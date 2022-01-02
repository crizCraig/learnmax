import argparse
import json
import math
import os
import time
from copy import copy
from typing import Tuple, List, Optional

import cv2
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from gym import Env, ObservationWrapper
from loguru import logger as log
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.datamodules.experience_source import Experience
from pl_bolts.models.rl.common.gym_wrappers import (
    make_environment, ImageToPyTorch, ProcessFrame84, FireResetEnv, MaxAndSkipEnv, ScaledFloatFrame)
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pl_bolts.utils import _OPENCV_AVAILABLE
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn import functional as F

from learn_max import dvq
from learn_max.agent import LearnMaxAgent, AgentState
from learn_max.dvq.model.deepmind_enc_dec import ResBlock
from learn_max.utils import topk_interesting, _init_weights, get_batch_vars, get_date_str, get_action_states
from learn_max.constants import SAVE_DIR, SEED, DEBUGGING, DATE_STR, ROOT_DIR, RUN_ID
from learn_max.dvq.vqvae import VQVAE, DecayLR, DecayTemperature, RampBeta
from learn_max.mingpt.lr_decay import GptWarmupCosineLearningRateDecay
from learn_max.mingpt.model import GPT


class LearnMax(pl.LightningModule):
    def __init__(
            self,
            # dvq_embedding_dim: int = 4410,  # length of embedding vectors output by dvq to transformers

            embedding_dim: int = None,
            num_embeddings: int = None,  # Number of possible discrete states shared between dvq and gpt

            # dvq args - dvq = deep vector quantization
            dvq_n_hid: int = 64,  # number of channels controlling the size of the model
            # dvq_num_embeddings: int = 512,   # now num_embeddings,  vocabulary size; number of possible discrete states
            dvq_loss_flavor: str = 'l2',  # `l2` or `logit_laplace`
            dvq_input_channels: int = 3,  # 3 for RGB
            dvq_enc_dec_flavor: str = 'deepmind',  # Deepmind VQVAE or OpenAI Dall-E dVAE
            dvq_vq_flavor: str = 'vqvae',  # `vqvae` or `gumbel`
            dvq_quantize_proj: int = None,
            dvq_checkpoint: str = None,  # pretrained dvq checkpoint
            dvq_enable_kmeans: bool = True,  # whether to train the clusters iteratively with kmeans

            # mingpt model definition args
            # size of the vocabulary (number of possible tokens) -
            #  64 for Shakespeare
            #  8,192 in DALL·E images
            #  16,384 for DALL·E words
            # gpt_vocab_size: int = 64,  # now num_embeddings

            # length of the model's context window in time
            gpt_block_size: int = 80,
            gpt_n_layer: int = 8,  # depth of the model; number of Transformer blocks in sequence
            gpt_n_head: int = 10,  # number of heads in each multi-head attention inside each Transformer block
            gpt_batch_size: int = 16,  # with such large embeddings (4410) needs to be small now to fit on rtx 2080

            # mingpt model optimization args
            training_gpt: bool = None,  # whether to train gpt
            gpt_learning_rate: float = 3e-4,  # the base learning rate of the model
            gpt_weight_decay: float = 0.1,  # amount of regularizing L2 weight decay on MatMul ops
            gpt_betas: Tuple[float, float] = (0.9, 0.95),  # momentum terms (betas) for the Adam optimizer
            gpt_embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
            gpt_resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
            gpt_attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix

            # Whether to use embeddings or indexes as input to first layer. With characters, we learn an embedding
            # from the ascii index, but with dvq inputs, the embeddings are high dimensional and
            # semantically meaningful so we optionally use those as input.
            # Learning the embedding may still be useful though as it can be jointly trained with the position
            # (and perhaps action embedding) which are summed as input to the transformer.
            # update: Just passing the dvq embedding does not work, possibly due to summing with the position.
            # Need to try concatenating the dvq embedding with a learned embedding.
            gpt_input_embed: bool = True,

            # RL/sensorimotor type stuff
            avg_reward_len: int = 100,  # how many episodes to take into account when calculating the avg reward
            epoch_len: int = 1000,  # how many batches before pseudo epoch
            gamma: float = 1,  # discount factor - only used for evaluation metrics right now
            n_steps: int = 1,  # number of steps to return from each environment at once
            replay_size: int = 30_000,  # number of steps in the replay buffer - tune w/ system memory
            env_id: str = 'MontezumaRevenge-v0',  # gym environment tag
            warm_start_size: int = 10_000,  # how many samples do we use to fill our buffer at the start of training
            batches_per_epoch: int = 10_000,  # number of batches per pseudo (RL) epoch

            # Standard stuff
            num_workers: int = 0,  # data loader workers - pycharm has issues debugging these. also gen batch requires NN for action so can't use > 0 at runtime either yet
            data_dir: str = SAVE_DIR,  # place to save tfboard logs and checkpoints
            batch_size: int = 32,  # do we have a batch size? or are gpt and dvq batch sizes adequate?
            # checkpoint: str = None, # Checkpoint to restore from

            single_token2: bool = False,
    ):
        """
        Maximizing learning by predicting interesting actions

        Use a deep vector quantizer to compress high dimensional continuous inputs into categorical inputs which can
        be more easily predicted by a transformer. We define a 'safe' objective as one that lower bounds safety to
        cases that avoid stagnation, destruction, or termination of learning by maximizing learning over time. We
        maximize learning over time by predicting the sequence of states and actions that have uncertainty. How much
        uncertainty is an open question but for humans it appears to be 50-70% [1]. Long term predictions will likely
        need to be made by a hierarchy of transformers which each predict "salient" events from the level below.
        Salient events can be defined by some distance threshold surpassed in a sliding window of sequential events.

        Additionally, we may be able to predict further out, by limiting the auto-regression to the most likely paths.

        TODO:
        Talk about how this solves all concrete problems in AI safety, but leaves replacement scenarios (i.e.
        grey goo, computronium, borg) as pathological cases. We do however avoid stagnation and destruction scenarios
        without needing to coordinate the world around pursuing safe AGI. Instead pursuing the most capable AI is all
        that is needed for the above, given that capability and maximizing learning are aligned.

        [1] https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036399
        """
        super().__init__()
        log.info(f'{num_workers=}')
        self.is_single_token2 = single_token2
        default_num_embeddings, default_embedding_dim = get_default_embeddings(self.is_single_token2)

        if num_embeddings is None:
            num_embeddings, embedding_dim = default_num_embeddings, default_embedding_dim
        # Hyperparameters
        self.dvq_embedding_dim = embedding_dim  # size of the embedding vector representing a cluster of embeddings
        self.dvq_n_hid = dvq_n_hid
        self.dvq_num_embeddings = num_embeddings
        self.dvq_loss_flavor = dvq_loss_flavor
        self.dvq_input_channels = dvq_input_channels
        self.dvq_enc_dec_flavor = dvq_enc_dec_flavor
        self.dvq_vq_flavor = dvq_vq_flavor
        self.dvq_checkpoint = dvq_checkpoint
        self.dvq_ready = False  #  Whether the dvq has enough data to train a batch
        if training_gpt is None:
            self.training_gpt = True if self.dvq_checkpoint is not None else False
        else:
            self.training_gpt = training_gpt

        # the "width" of the model (embedding_dim), number of channels in each Transformer
        self.gpt_input_embedding_dim = embedding_dim
        self.gpt_block_size = gpt_block_size  # Size of the temporal window
        self.gpt_batch_size = gpt_batch_size
        self.gpt_n_layer = gpt_n_layer
        self.gpt_n_head = gpt_n_head
        self.gpt_learning_rate = gpt_learning_rate
        self.gpt_weight_decay = gpt_weight_decay
        self.gpt_betas = gpt_betas
        self.gpt_embd_pdrop = gpt_embd_pdrop
        self.gpt_resid_pdrop = gpt_resid_pdrop
        self.gpt_attn_pdrop = gpt_attn_pdrop

        self.avg_reward_len = avg_reward_len
        self.epoch_len = epoch_len
        self.gamma = gamma
        self.n_steps = n_steps
        self.replay_size = replay_size
        self.env_id = env_id
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch

        # RL / sensorimotor stuff
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.batch_size = gpt_batch_size if self.training_gpt else batch_size

        self.total_steps = 0
        self.total_rewards = []
        self.total_episode_steps = []

        self.save_hyperparameters()

        self.dvq_optimizer = None
        self.gpt_optimizer = None

        # We use multiple optimizers so need to manually backward each one separately
        self.automatic_optimization = False

        self.agent_state = None  # Most recent internal and external state

        def make_env(_env_id):
            _env = gym.make(_env_id)
            # _env = MaxAndSkipEnv(_env)
            _env = FireResetEnv(_env)

            # These wrappers also convert to grayscale, which I think was done
            # to be able to stack frames, but we should probably not do.
            _env = ProcessFrame84Color(_env)
            _env = ImageToPyTorch(_env)

            # We don't want to stack frames, the transformer will do sequential inference
            # _env = BufferWrapper(_env, 4)

            _env = ScaledFloatFrame(_env)
            return _env

        if 'USE_STANDARD_ENV_WRAPPERS' in os.environ:
            # Thinking these slow things down a lot, but at what benefit?!
            # These wrappers also convert to grayscale, which I think was done
            # to be able to stack frames, but we should probably not do.
            self.env = self.make_environment(env_id)
            self.env.seed(SEED)
            self.test_env = self.make_environment(env_id)
        else:
            self.env = make_env(env_id)
            self.env.seed(SEED)
            self.test_env = make_env(env_id)

        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # This action embedding will be concatenated with the dvq output. The transformer
        #  will then enumerate all actions for all possible dvq tokens for n_actions * n_embd total possible
        #  outputs from the softmax. This allows selecting actions with the known next state.
        # Saliency levels above that will have abstract actions, however, we will use the same
        # dvq states with actions as keypoints. Saliency levels are combined into the same transformer by
        # concating a context token to the input.
        # The transformer will know what saliency level next z,a from this prefix. This saliency token will represent
        # the salience level and the step within that level (so could perhaps be considered two tokens).

        # OLD IDEA FROM WHICH THIS CAME - NOT SCALABLE DUE TO MANY TRANSFORMERS
        # Abstract actions will live in combined (z,a) tokens along with abstract states. These tokens will be generated
        # by dvq's which take in z,a's from below them.
        #
        # Low-level actions will therefore require simply taking the postfix from the txf predicted next token.

        # Only one dvq is needed in order to transform concrete into abstract states.
        # If a novel state is encountered, we must look at the salience sequence above us to determine if the state
        # was unpredicted in that level as well. If so, we continue up a saliency level, until reaching the cieling
        # saliency. https://drive.google.com/file/d/1HE1zsZnw41A7l1Hn5VEgyS0b59fgLYxu
        # The image linked is above except that the reserved cieling token will not be reserved.
        # The saliency decoder can be just an MLP with 2 output softmax's which
        # maps a large set of dvq outputs to a given saliency level + saliency step.
        # We don't start a new saliency level until the expected deviation of the transformer output decreases to a
        # point where have a good grasp of the probabilities at the current saliency level.
        # Basically we don't make a new level until unknown unknowns are reduced to known unknowns. I.e. we
        # know if something is inherently noisy / evenly distributed vs unexplored at a given saliency level.
        # Once the expected deviation of transformer output probabilities goes down on average at a given saliency
        # level, but THEN we receive a highly surprising state (as determined via dvq reconstruction loss, low predicted
        # probability by gpt, mahalonbis distance of the centroid from pen-ultimate dvq layer,
        # or another uncertainty measure) only THEN do we create a _new_ salient state.
        # This _should_ have the effect of compressing saliency levels over time as we become less surprised by
        # our higher level plans and repeat them consistently. If this happens, we should notice the keypoint state
        # for the saliency level above has been reached without novelty and skip that state in the input to the
        # saliency level above for future training.
        # So when we start training, everything will have the same saliency context until uncertainty goes down.
        # Also, since the saliency decoder predicts an integer via the softmax, we need to map the integer to an embedding
        # with learned embedding.


        # TODO: In order to concatenate this with the state embedding, shrink the state embedding by the size of the
        #   action embedding. Then make sure the softmax size is n_actions * n_state_embed
        # self.action_emb = nn.Embedding(self.n_actions, action_embedding_dim)

        self.apply(_init_weights)  # Should only be action embeddings, other weights are in dvq and gpt

        # RL type tracking metrics
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.batch_states = []
        self.batch_actions = []

        if self.is_single_token2:
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_embeddings, embedding_dim=self.dvq_embedding_dim,
                             loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                             enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor,
                             quantize_proj=dvq_quantize_proj,
                             is_single_token2=self.is_single_token2, enable_kmeans=dvq_enable_kmeans)
        else:
            # TODO: Need to test that distance between clusters is further than avg distance between points in a cluster
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_embeddings, embedding_dim=self.dvq_embedding_dim,
                             loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                             enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor, quantize_proj=None,
                             is_single_token2=self.is_single_token2)

        self.gpt = GPT(vocab_size=num_embeddings * self.num_actions, block_size=gpt_block_size, n_layer=gpt_n_layer,
                       input_embedding_dim=self.gpt_input_embedding_dim, n_head=gpt_n_head,
                       learning_rate=gpt_learning_rate, weight_decay=gpt_weight_decay, betas=gpt_betas,
                       embd_pdrop=gpt_embd_pdrop, resid_pdrop=gpt_resid_pdrop, attn_pdrop=gpt_attn_pdrop,
                       should_input_embed=gpt_input_embed, num_actions=self.num_actions)

        self.agent = LearnMaxAgent(model=self, num_actions=self.env.action_space.n)
        self.reset()

    def reset(self):
        self.agent_state = self.get_agent_state(self.env.reset())
        return self.agent_state

    def get_agent_state(self, state):
        if not isinstance(state, list):
            state = [state]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state), device=self.device)  # causes issues when num_workers > 0
        x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.dvq.forward(state)
        agent_state = AgentState(state=state, dvq_x=x, dvq_x_hat=x_hat, dvq_z_q_emb=z_q_emb, dvq_z_q_flat=z_q_flat,
                                 dvq_latent_loss=latent_loss, dvq_recon_loss=recon_loss, dvq_loss=dvq_loss,
                                 dvq_z_q_ind=z_q_ind)
        return agent_state

    # def run_n_episodes(self, env, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
    #     """
    #     Carries out N episodes of the environment with the current agent
    #
    #     Args:
    #         env: environment to use, either train environment or test environment
    #         n_epsiodes: number of episodes to run
    #         epsilon: epsilon value for DQN agent
    #     """
    #     total_rewards = []
    #
    #     for _ in range(n_epsiodes):
    #         episode_state = env.reset()
    #         done = False
    #         episode_reward = 0
    #
    #         while not done:
    #             self.agent.epsilon = epsilon
    #             action, agent_state = self.agent(episode_state, self.device)
    #             next_state, reward, done, _ = env.step(action[0])
    #             episode_state = next_state
    #             episode_reward += reward
    #
    #         total_rewards.append(episode_reward)
    #
    #     return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            agent_state = self.reset()
            for _ in range(warm_start):
                action = self.agent.get_action(agent_state, self.device)
                next_state, reward, done, info = self.env.step(action[0])
                next_agent_state = self.get_agent_state(next_state)
                exp = Experience(state=self.agent_state, action=action[0], reward=reward, done=done,
                                 new_state=next_agent_state)
                self.buffer.append(exp)
                self.agent_state = next_agent_state
                if done:
                    self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """

        # Real forward is in agent currently.
        output = self.net(x)
        return output

    def train_batch(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        self.profile_cuda_usage_tensors()

        actions, agent_states, dones, new_states, rewards, states = (None, None, None, None, None, None)

        while True:
            if self.total_steps == 0 or 'OVERFIT' not in os.environ:
                # TODO: Do more than one action per batch, but not so much that we can't learn from recent experience
                #   i.e. allow learning within the episode
                # TODO: Move this tuple into a @dataclass
                (states, actions, rewards, dones, new_states, agent_states, next_agent_states, episode_steps,
                    episode_reward) = self._take_action_and_sample(episode_reward, episode_steps)

                # actions, agent_states, dones, new_states, rewards, states, episode_steps, episode_reward = \
            if dones is None:
                log.error('dones is None, what is going on??? - trying to continue')
                time.sleep(5)
                continue
            for idx, _ in enumerate(dones):
                # Lightning wants tensors, numpy arrays, numbers, dicts or lists in batch
                if isinstance(agent_states[idx], AgentState):
                    # agents_states only has a batch dimension, not window size
                    agent_states = [_.dict() for _ in agent_states]
                else:
                    # agents_states has batch and window size dimensions
                    agent_states = [_.dict() for _ in agent_states[idx]]
                # print('states', states[idx][0][0])
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx], agent_states

            self.total_steps += 1

            # if self.total_steps % 1000 == 0:
            #     from guppy import hpy
            #     h = hpy()
            #     print('heap stats', h.heap())

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def _take_action_and_sample(self, episode_reward, episode_steps):
        episode_reward, episode_steps, is_done = self._take_action(episode_reward, episode_steps)
        if is_done:
            self.done_episodes += 1
            self.total_rewards.append(episode_reward)
            self.total_episode_steps.append(episode_steps)
            self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
            self.reset()
            episode_steps = 0
            episode_reward = 0
        if self.training_gpt:
            # Sample sequentially from replay buffer
            # If training GPT, we need to return the dvq state indexes as well to be used in on_train_batch_end
            #   in lr_decay _, y = batch
            states, actions, rewards, dones, new_states, agent_states, next_agent_states = \
                self.sample_latest_sequential()
            # agent_states, actions, rewards, dones, new_agent_states = self.sample_latest_sequential()
        else:
            # Sample randomly from replay buffer
            agent_states, actions, rewards, dones, new_agent_states = self.buffer.sample(
                self.batch_size)  # TODO: Change to dvq_batch_size?
            raise NotImplementedError('Get DVQ training to work with new agent_state protocol, old code below')
            states = np.array([x for x in states[:, 0]])
            new_states, agent_states = new_states[:, 0], new_states[:, 1]
            new_states = np.array([x for x in new_states])
        return (states, actions, rewards, dones, new_states, agent_states, next_agent_states, episode_steps,
                episode_reward)

    def _take_action(self, episode_reward, episode_steps):
        action = self.agent.get_action(self.agent_state, self.device)
        next_state, r, is_done, _ = self.env.step(action[0])
        next_agent_state = self.get_agent_state(next_state)
        episode_reward += r
        episode_steps += 1
        exp = Experience(state=self.agent_state, action=action[0], reward=r, done=is_done, new_state=next_agent_state)
        self.buffer.append(exp)
        self.agent_state = next_agent_state
        return episode_reward, episode_steps, is_done

    def profile_cuda_usage_tensors(self):
        import torch
        import gc
        log.info('Profiling torch objects')
        log.info('-----------------------')
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    log.info(type(obj), obj.size())
            except:
                pass
        log.info('Done profiling torch objects')
        log.info('-----------------------')

    def sample_latest_sequential(self):
        """Get batch size X gpt_block_size windows for GPT"""
        start_indices = np.random.choice(len(self.buffer) - self.gpt_block_size + 1, self.gpt_batch_size, replace=False)
        ret_states = []
        ret_agent_states = []
        ret_actions = []
        ret_rewards = []
        ret_dones = []
        ret_next_states = []
        ret_next_agent_states = []
        block_idx = np.array(list(range(self.gpt_block_size)))
        for s in start_indices:
            # TODO: pad with -100 when last index? We just avoid by making sure we sample early enough in the buffer
            indices = s + block_idx
            agent_states, actions, rewards, dones, next_agent_states = zip(*[self.buffer.buffer[idx] for idx in indices])
            states = agent_states['state']
            ret_states.append([s[0] for s in states])
            ret_agent_states.append(agent_states)
            ret_actions.append(actions)
            ret_rewards.append(rewards)
            ret_dones.append(dones)
            next_states = next_agent_states['state']
            ret_next_states.append(next_states)
            ret_next_agent_states.append(next_agent_states)
        # TODO: Speed this up with numba?
        # TODO: Just put everything in the agent_states, next_agent_states dict's
        ret = (
            np.array(ret_states),
            np.array(ret_actions),
            np.array(ret_rewards, dtype=np.float32),
            np.array(ret_dones, dtype=bool),
            np.array(ret_next_states),
            ret_agent_states,
            ret_next_agent_states,
        )
        return ret

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Calculates loss based on the minibatch received

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        # loss = dqn_loss(batch, self.net, self.target_net, self.gamma)

        # TODO: We should train the dvq for a while, or load a checkpoint. Then train the GPT for a while on the learned
        #   tokens. The alt. would be to train everything in parallel which would be cool, but the transformer would
        #   need to adjust to the distribution shift, i.e. token centroids changing. The DVQ could always be trained
        #   online with the replay buffer. Keeping it frozen for the gpt would be like the target network in DQN.
        #   The target network was needed because the loss was dependent on it. Here it's the input that's dependent
        #   on the other network. We'll just have to see what works.

        if self.training_gpt:
            self.dvq.set_do_kmeans(False)
            gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y, s = get_batch_vars(batch, return_agent_state=True, populate_gpt=True)
            # print('gptx', gpt_x[0][0])
            # Train gpt on dvq tokens shifted for prediction

            gpt_ret = self.gpt.training_step(batch=(gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y), batch_idx=batch_idx)

            if batch_idx % 1000 == 0:
                self.viz_gpt(gpt_ret, state=s, batch_idx=batch_idx)

            # We use multiple optimizers so need to manually backward each one separately
            self.gpt_optimizer.zero_grad()
            self.manual_backward(gpt_ret['loss'])  # , self.gpt_optimizer)

            # Required due to https://github.com/PyTorchLightning/pytorch-lightning/issues/7698
            torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), 1.0)
            self.gpt_optimizer.step()

            loss = gpt_ret['loss']
            # TODO: For distributed training?
            #  Although use_dp is not available now, would need self.trainer._accelerator_connector.strategy
            # if self.trainer.use_dp or self.trainer.use_ddp2:
            #     loss = loss.unsqueeze(0)
        else:
            latent_loss, dvq_loss, recon_loss = self._train_step_dvq(batch, batch_idx)
            loss = dvq_loss
            self.log_dict({
                "dvq_recon_loss": recon_loss,
                "dvq_latent_loss": latent_loss, })

        self.log_dict({
            "total_reward": self.total_rewards[-1] if self.total_rewards else 0,
            "avg_reward": self.avg_rewards,
            "train_loss": loss,
            "episodes": self.done_episodes,
            "episode_steps": self.total_episode_steps[-1] if self.total_episode_steps else 0,
        })

        # return loss
        # return OrderedDict({
        #     "loss": loss,
        #     "avg_reward": self.avg_rewards,
        # })

    def viz_all_dvq_clusters(self):
        self.cuda()
        wandb.init(entity='crizcraig', mode='disabled')
        n = self.dvq_num_embeddings
        width = int(n ** 0.5)
        col_i = 0
        row = []
        rows = []
        for i in range(n):
            _i = 3070 if 'FAKE_Z_Q_EMB' in os.environ else i
            emb = self.dvq.quantizer.embed_code(torch.tensor([_i]).cuda())  # TODO: Speed up with batching
            img_t = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)
            img_t = img_t.squeeze().permute(1, 2, 0).clamp(0, 1)
            row.append(img_t.detach().cpu().numpy())
            col_i += 1
            if col_i > width:
                col_i = 0
                rows.append(np.concatenate(row, axis=1))
                row = []
        last_row = np.concatenate(row, axis=1)
        last_row_remain = rows[-1].shape[1] - last_row.shape[1]
        last_row_padded = np.concatenate((last_row, np.zeros((rows[-1].shape[0], last_row_remain, 3))), axis=1)
        rows.append(last_row_padded)
        x_hat = np.array(rows).reshape((84 * width, -1, 3))
        im = Image.fromarray(np.uint8(x_hat * 255))
        im.show()
        im.save(f'{ROOT_DIR}/images/all_dvq_clusters_{DATE_STR}.png')

    def viz_dvq(self):
        self.cuda()
        wandb.init(entity='crizcraig', mode='disabled')

        test_loader = self.test_dataloader()
        xl = next(iter(test_loader))[0].cuda()
        x = xl.cuda().reshape(-1, 3, 84, 84)
        # TODO: List => Tensor
        # x = [t.cuda() for t in xl]
        # x_hat = [t['dvq_x_hat'].cuda() for t in x]
        # loss, recon_loss, latent_loss, x_hat = model.dvq.training_step(x, 0)
        # 80,3,84,84
        x2, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.dvq.forward(x)
        # states, actions, rewards, dones, new_states = next(iter(test_loader))
        # x = torch.cat([states, new_states])
        num_cols = 20
        xcols = torch.cat([x[:num_cols], x_hat[:num_cols]], axis=2)  # side by side x_pre and xhat
        xrows = torch.cat([xcols[i] for i in range(num_cols)], axis=2)

        # TODO: Just save image with PIL here
        plt.figure(figsize=(num_cols, 5))
        plt.imshow((xrows.data.cpu().permute(1, 2, 0)).clamp(0, 1))
        plt.axis('off')
        plt.show()

        plt.show()

    def viz_gpt(self, gpt_ret, state, batch_idx):
        batches_to_visualize = 1
        for b_i in range(batches_to_visualize):
            num_imgs_to_viz = 10
            top_n = 3
            imgs = state[b_i][:num_imgs_to_viz]  # Check start of sequence

            logits = gpt_ret['logits'].detach().cpu().numpy()[b_i][:num_imgs_to_viz]

            # dvq states are action agnostic, so get base state by dividing by num_actions
            target_idx = self.gpt.as_i_to_s_i(gpt_ret['target_idx'][b_i][:num_imgs_to_viz])

            emb = self.dvq.quantizer.embed_code(target_idx)
            imgs_target = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)

            # top n predicted next embedding indexes for each img - TODO: Use torch.topk here which is already sorted
            top_n_idxs = np.argpartition(logits, -top_n)[:, -top_n:]  # top n idxs unsorted
            top_n_idxs_idxs_sorted = np.argsort(np.take_along_axis(-logits, top_n_idxs, axis=1))  # -logits => sort desc
            top_n_idxs = np.take_along_axis(top_n_idxs, top_n_idxs_idxs_sorted, axis=1)  # top n idxs sorted desc
            top_n_idxs = torch.Tensor(top_n_idxs).int()

            # get dvq embeddings from indexes
            target_idx_state = self.gpt.as_i_to_s_i(top_n_idxs.to(target_idx.device))

            emb_hat = self.dvq.quantizer.embed_code(target_idx_state)

            # decode predictions into images
            imgs_hat = self.dvq.decode_flat(emb_hat, output_proj=self.dvq.quantizer.output_proj)
            sn = len(imgs_hat.size())
            imgs_hat = imgs_hat.permute(*list(range(sn-3)), sn-2, sn-1, sn-3).clamp(0, 1)
            imgs_display = []
            for img_i, img in enumerate(imgs):
                img = img.permute(1, 2, 0).cpu().numpy()
                img_target = imgs_target[img_i].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
                blank = 0 * np.zeros_like(img)
                img_hat_row = []
                for ti in range(top_n):
                    img_hat = imgs_hat[img_i][ti].clamp(0, 1)
                    img_hat_row.append(img_hat.detach().cpu().numpy())
                left_img = np.concatenate([blank, img, blank])
                mid_img = np.concatenate(img_hat_row)
                right_img = np.concatenate([blank, img_target, blank])
                imgs_display.append(np.concatenate([left_img, mid_img, right_img], axis=1))
                imgs_display.append(0.5 * np.ones((5, *imgs_display[-1].shape[1:])))  # border

            imgs_display = np.concatenate(imgs_display)
            im = Image.fromarray(np.uint8(imgs_display * 255))
            # im.show()
            filename = f'{ROOT_DIR}/images/viz_gpt_{get_date_str()}_r_{RUN_ID}_e_{self.current_epoch}_b_{batch_idx}.png'
            log.info(f'Saving gpt viz to {filename}')
            im.save(filename)
            wandb.log({'train/gpt-viz': [wandb.Image(im)]})

            # fig1 = plt.figure(figsize=(num_imgs_per_row, 1))
            # ax1 = fig1.add_subplot(111)
            # ax1.imshow(imgs_display, interpolation='none')
            # plt.show()
            # TODO: Visualize image with action, cluster index + distance, probability, uncertainty
            #   Make sure you're extracting the logits correctly for the image
            #   Look at disappear and teleport images bad clustering cases.

    def _train_step_dvq(self, batch, batch_idx):
        self.dvq.set_do_kmeans(True)

        _, agent_state = get_batch_vars(batch, return_agent_state=True)

        # TODO: No need to call training_step again, just need to get the average loss from the batch.
        dvq_loss = torch.mean(torch.Tensor([a["dvq_loss"].mean() for a in agent_state]))
        recon_loss = torch.mean(torch.Tensor([a["recon_loss"].mean() for a in agent_state]))
        latent_loss = torch.mean(torch.Tensor([a["latent_loss"].mean() for a in agent_state]))

        # dvq_loss, recon_loss, latent_loss, x_hat = self.dvq.training_step(batch, batch_idx)

        # We use multiple optimizers so need to manually backward each one separately
        self.dvq_optimizer.zero_grad()
        self.manual_backward(dvq_loss)
        self.dvq_optimizer.step()
        loss = dvq_loss
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
        return latent_loss, loss, recon_loss

    # def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    #     """Evaluate the agent for 10 episodes"""
    #     test_reward = self.run_n_episodes(self.test_env, 1, 0)
    #     avg_reward = sum(test_reward) / len(test_reward)
    #     return {"test_reward": avg_reward}
    #
    # def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
    #     """Log the avg of the test results"""
    #     rewards = [x["test_reward"] for x in outputs]
    #     avg_reward = sum(rewards) / len(rewards)
    #     self.log("avg_test_reward", avg_reward)
    #     return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        self.dvq_optimizer = self.dvq.configure_optimizers()
        self.gpt_optimizer = self.gpt.configure_optimizers()
        return [self.dvq_optimizer, self.gpt_optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        # Also, need to disable kmeans in an object variable and avoid backpropping elsewhere
        if self.training_gpt:
            for prop_name in dir(self.dvq):
                prop = getattr(self.dvq, prop_name)
                if isinstance(prop, nn.Module):
                    prop.zero_grad()
            batch_size = self.gpt_batch_size
            self.dvq.set_do_kmeans(False)
            log.info(f'Not training dvq so setting warm start size to batch size. '
                     f'Was {self.warm_start_size}, now  is {self.batch_size}')
            self.warm_start_size = self.gpt_batch_size * self.gpt_block_size
        else:
            batch_size = self.batch_size  # Already set in __init__ but in case we toggle training gpt, set here too
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        log.info(f'Populating replay buffer with {self.warm_start_size} experiences...')
        self.populate(self.warm_start_size)
        log.info(f'...finished populating replay buffer')

        # train_batch calls the model to get the next action, so each worker has a copy of the model!
        self.dataset = ExperienceSourceDataset(self.train_batch)

        return DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.num_workers,
                          drop_last=True, pin_memory=False)  # Images on GPU already so pin_memory raises exception

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()

    @staticmethod
    def make_environment(env_name: str, seed: Optional[int] = None) -> Env:
        """
        Initialise gym  environment

        Args:
            env_name: environment name or tag
            seed: value to seed the environment RNG for reproducibility

        Returns:
            gym environment
        """
        env = make_environment(env_name)

        if seed:
            env.seed(seed)

        return env

    @staticmethod
    def add_model_specific_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
        arg_parser.add_argument("--dvq_quantize_proj", type=int, default=None)
        arg_parser.add_argument("--num_gpus", type=int, default=1)
        arg_parser.add_argument("--single_token2", action='store_true', default=False)
        arg_parser.add_argument('--num_workers', type=int, default=None, help="number of workers for dataloading")
        arg_parser.add_argument('--viz_dvq', type=str, help="visualize dvq images", default=None)
        arg_parser.add_argument('--viz_all_dvq_clusters', type=str, help="visualize all dvq clusters", default=None)
        arg_parser.add_argument('--dvq_checkpoint', type=str, help="Checkpoint to restore", default=None)
        arg_parser.add_argument('--gpt_batch_size', type=int, help="GPT batch size", default=8)
        arg_parser.add_argument('--gpt_block_size', type=int, default=40,
                            help="block size for the model (length of window of context)")
        return arg_parser

    @staticmethod
    def add_reinforcement_learning_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note:
            These params are fine tuned for Pong env.

        Args:
            arg_parser: parent parser
        """
        arg_parser.add_argument(
            "--replay_size",
            type=int,
            default=30_000,  # Tune with system memory
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=int(os.getenv('WARM_START', 10_000)),
            help="how many samples do we use to fill our buffer at the start of training",
        )

        arg_parser.add_argument("--batches_per_epoch", type=int, default=10_000, help="number of batches per pseudo (RL) epoch")
        arg_parser.add_argument("--env_id", type=str, help="gym environment tag", default='MontezumaRevenge-v0')
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        return arg_parser

    def tree_search(self, beam_width=4, num_levels=10):
        """
        Need to keep track of action taken at start of search path, total uncertainty, and path nodes.

        Avoiding discount factor for now as this already happens due to limited prediction horizon. Also unclear if
            discount factor is fundamentally good: https://stats.stackexchange.com/questions/221402/

        Beam width should be batch size unless transformers combine batch + sequence trunk with the torch.tril mask

        We want total path interestingness (deviation) in the 50-70th percentile of all path costs.

        We should keep track of total interestingness along all paths searched, then go the 50-70th percentile of
        current paths.

        deviation = [
              0    1    2    3
        0    [0.1, 0.2, 0.3, 0.4],
        1    [0.2, 0.9, 0.1, 0.1],
        2    [0.1, 0.2, 0.1, 0.1],
        3    [0.2, 0.2, 0.3, 0.1],]

        #     0.6  1.5  0.8  0.7

        q = 0.2=(0,1), 0.3=(0,2)

        totals = [0.1, 1.1, 0.4, 0.4]

        q = 0.1=(2,2), 0.1=(1,3)

        totals = [0.1, 1.1, 0.5, 0.5]

        q = 0.1=(0,2), 0.1=(1,3)

        totals = [0.1, 1.1, 0.8, 0.6]

        * take the transformer's input sequence of z embeddings and softmax entropy as input to tree search
          (note that entropy is the proxy we are using for uncertainty)
        * we are predicting action-states, so given state s0, predict num_actions * num_z
            states which represent taking some action and landing in some state
        * TODO: We need to think about backtracking to most promising forks since we aren't exhaustively exploring
            MCTS, A*, and beam search should have ideas here
        * add path uncertainties to _all_ encountered sorted path uncertainties
            * try naive unoptimal way first though since we don't have that many branches)
            * If we need to optimize, use TORCH.SEARCHSORTED
                * insert new uncertainties with torch.cat - but also keep track of the associated sequence index in another
                  (unsorted) dimension of the the uncertainties pool
            * sequences just get appended to and are not sorted, so sequence index can be static.
        * get topk_interesting(action-states, k) actions - (nb. interesting is defined as within 50-100pct
          action entropy) - however we may need to do 50-100pct to avoid state starvation
          there will be some trajectories at high saliency levels that are the most interesting. we hope that those
          trajectories remain most interesting for some amount of time so that we're not switching strats to often.
        * Now get the most likely states arising from those actions according to the softmax output and add those to
          the search tree.
        * feed the corresponding a,z embeddings at those indexes back into the transformer at the end of the
          current sequence (we are simulating the env in multiple future branches)
        * get new states and uncertainties for the next level of the search and add them to the sorted list according to above
        * when we build up a pool of transformer i/o that reaches some desired max size in GPU memory, get the most
          interesting trajectory and return the first action of that trajectory.
        * we can now throw away the search tree
        * add the new experience to the training batch for the transformer as normal
        * repeat the above search for the next action
        * in tandem, store new observations from the environment in a batch to train the dvq online. this will mean
          that the embedding centroids will change and that the transformer will be associating moving centroids
          (different points) with the same output indexes over time. if this is a problem, we will need to train the
          dvq on bigger batches and try to pretrain it with some videos
          https://arxiv.org/abs/1805.11592 or pretrained agents perhaps
          (TODO: Ask Aditya Ramesh about this as he tried training jointly with DALL-E and didn't notice improvement,
           We don't need improvement, we just need this to deal with shifting input distributions.
           # TODO: Short term saliency layer(s) for working memory. Could use high learning rate or multiple overlapping
           #    layers.


        Saliency levels

        Let’s say you have only the first saliency level and you get a sequence of predictable states followed by a less
        predicted state (say jumping onto the ladder whereas usually you fall off or stay on the start platform).

        (We can use the softmax prob or reconstruction error as a proxy for how predictable the state was.)

        Now the single state novelty is higher and the uncertainty is low enough (on average across a recent sliding window)
        that we can create a novel state in the saliency level above.
        Initially the saliency context is just the first step. Now it will be the state where you are on the platform.

        This context token would allow forming a different plan (even if it’s longer than the transformer’s prediction
        window) depending on what the higher level dictates at that point.

        A problem with uncertainty-defined saliency is that uncertainty will change and perhaps salience should not.
        Therefore, it may be better to define salience as a KL divergence in the action-state softmax across a rollout.
        Basically the idea is that salient states are defined by unlocking possibilities.
        So when pananma joe is just jumping around on the initial platform, the state possibilities for the next
        4 or 5 actions are largely the same. But when he jumps to the ladder or down to the bottom, the states reachable
        from there are very different and so the softmax distribution changes quite a bit.

        It may be that uncertainty measured via entropy and possibilities measured through KL divergence are largely
        the same. I think the important point is that we look at the next n actions in the tree search to determine
        how much uncertainty or possibility there is and create a new salient state from that instead of just
        looking a single state. This way if it takes a few actions for panama joe to just end up at the same place,
        (i.e. there's a cycle) then the normalized probability of future states will be less diverse and therefore
        less interesting than if joe jumps to the ladder or down below.

        However, we should also check that the future possibilities are low enough entropy such that we aren't just
        totally unsure about the future and calling that a salient state.

        When we see saliency in the tree search (i.e. expanded low entropy possibilities), we should add a salient
        state to the training data at two levels: one where the salient state is the context,
        and one where the salient state is the level above and we are predicting salient states.

        ----------------------
        Design

        There should be context tokens: (previous salient), goal (next salient) state, and saliency level embedding
        added to each input token to the transformer. Also try splitting these across heads instead of adding so
        that we can pass zeroes to them when we don't have context and reuse the learning that takes place without
        context. We may also be able to reuse context when adding the tokens, but this (perhaps?) takes more network
        power even so.

        The salient states are just dvq outputs, i.e. z states at each saliency level, which allows the same transformer to be
        used for each saliency level. The difference is that, the context token will dictate skipping lower level states
        directly to the next salient state at that level.

        We need to stop adding levels when we run out of memory.

        I think we'll have to pass zeroes in for the next salient when we are starting out (or just don't add it).
        We only know the goal state when we've searched for the most interesting path and we know the next salient
        state on the way at the current level.

        Compression:
            If the next salient state is reached and it's no longer surprising, i.e. it's by far the most probable state
            and it has small reconstruction error, then we can train the level above to just skip this state as the
            current level can readily predict it. But we don't want this new transition to cause surprise that
            leads to a salient state above it. Checking to see if the predicted state (the old transition) was
            encountered at the lower level seems to complicated, so we can just avoid compression for now.
            **We do however need to keep track of when the goal state is reached and advance the saliency levels
              above when this happens**

        So we're going to have batches which represent search paths, and each iteration we'll be adding
        more search paths branching from the previous.

        We have three inputs z_q_flat, z_q_ind, and a

        Let's consider them as p

        So in one sense we need to push new elements onto the end of p like a deque. But in another sense we
        will be adding more branches and thus increasing p's size.

        This can be achieved by just stacking more trajectories on top of p after pushing new elements onto the deque.

        We need to keep track of which branches to add on to. Let's say p looks like this

        ABC
        ABD
        ABE

        And C results in two new states, D and E, then we should have

        ABCD
        ABCE

        in addition to whatever comes from ABD and ABE.

        So we should only append to the branch that spawned the state.



        IF we are searching greedily, then we just pursue the top k highest entropy actions. Saliency levels
        will mitigate the shortsightedness of this approach.

        HOWEVER, we could also add some monte-carlo rollouts in order to find states with delayed learning.

        For first level of the tree, only worry about one match (i.e. most recent action in env)
        For subsequent levels, the batch will represent different possible futures
        For ALL levels, only the last action in the window matters

        """

        # TODO:
        #   - If search finds termination, trim from tree. But what if all branches terminate???

        # Things to optimize if this is too slow (on GPU it's 100ms now with a 3090 so it's okay to play real time
        #   at least)
        #  1. optimize gpt forward
        #  2. do 3 to 5 step look ahead and rely on saliency for long term planning
        #  3. increase beam width for better quality look ahead (mostly parallel so is fast)
        log.info('Starting tree search')
        z_q_embed_branches = []
        z_q_ind_branches = []
        action_branches = []

        n = min(len(self.buffer), self.gpt_block_size)
        buffer = self.buffer.buffer
        # Populate gpt window with recent experience
        for i in reversed(range(n)):
            exp = buffer[i]
            z_q_embed_branches.append(exp.state.dvq_z_q_flat)
            z_q_ind_branches.append(exp.state.dvq_z_q_ind)
            action_branches.append(exp.action)

        # Unsqueeze so we have batch, window, embed dims, where batch=1, e.g. 1, 80, 4410
        z_q_embed_branches = torch.cat(z_q_embed_branches).unsqueeze(0)
        z_q_ind_branches = torch.cat(z_q_ind_branches).unsqueeze(0)
        action_branches = torch.tensor(action_branches).unsqueeze(0).to(self.device)
        entropy_branches = torch.tensor([0]).to(self.device)  # Total entropy of branch

        assert z_q_embed_branches.size()[0] == 1, 'We should only start with one trunk'
        action_entropy_path = None  # Declare for output
        for _ in range(num_levels-1):  # first level already populated
            # Move forward one step in the search tree
            log.debug('starting gpt forward')
            logits, expected_deviation = self.gpt.forward(embed=z_q_embed_branches[:, -self.gpt_block_size:],
                                                          idx=z_q_ind_branches[:, -self.gpt_block_size:],
                                                          actions=action_branches[:, -self.gpt_block_size:])
            log.debug('done with gpt forward')
            B, W, _ = z_q_embed_branches[:, -self.gpt_block_size:].size()  # batch size, window size

            # arrange so we can get entropy across each action
            logits = logits.reshape(B, W, -1, self.num_actions).transpose(-1, -2)
            logits = logits[:, -1:, ...]  # We only care about next step, so omit all but last part of window
            probs = F.softmax(logits, dim=-1)  # Probability of dvq clusters for actions
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # Entropy across actions

            # Add entropy gathered thus far in branches to entropy of most recent action
            entropy_path = (entropy_branches *
                            torch.ones((entropy.size()[-1], 1), device=self.device)).T.unsqueeze(1) + entropy

            actions, action_entropy_path, actions_flat = topk_interesting(entropy_path, k=beam_width)
            action_states = get_action_states(logits, actions_flat)

            # Branch out by copying beginning of path for each new action state and appending new action state to end
            new_z_q_ind_branches = []
            new_z_q_embed_branches = []
            new_action_branches = []
            new_entropy_branches = []
            log.debug('starting top action loop')
            for i, (bi, ai) in enumerate(actions):  # batches are branches
                # TODO: Vectorize this for loop for high beam sizes (>10?)

                # Get heads
                z_q_ind_head = z_q_ind_branches[bi]
                z_q_embed_head = z_q_embed_branches[bi]
                action_head = action_branches[bi]

                # Get tails
                z_q_ind_tail = action_states[i]
                z_q_embed_tail = self.dvq.quantizer.embed_code(z_q_ind_tail)
                action_tail = ai

                # Add head+tail to new branches
                new_z_q_ind_branches.append(torch.cat((z_q_ind_head, z_q_ind_tail.unsqueeze(0))))
                new_z_q_embed_branches.append(torch.cat((z_q_embed_head, z_q_embed_tail.unsqueeze(0))))
                new_action_branches.append(torch.cat((action_head, action_tail.unsqueeze(0))))
                new_entropy_branches.append(action_entropy_path[i])  # Already summed entropy along search path
            log.debug('done with top action loop')
            # Convert to tensor with window dim <= GPT block size
            z_q_ind_branches = torch.stack(new_z_q_ind_branches)
            z_q_embed_branches = torch.stack(new_z_q_embed_branches)
            action_branches = torch.stack(new_action_branches)
            entropy_branches = torch.stack(new_entropy_branches)

        log.info('Done with tree search')
        # Return action for highest entropy path
        ret = action_branches[torch.argmax(action_entropy_path)][1]   # 0th action has already been taken
        # TODO: Return trajectory of actions as gpt forward is too slow to just do one action at at time.
        return [ret]


    # Perform gradient clipping on gradients associated with gpt (optimizer_idx=1)
    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
    #     if optimizer_idx == 1:
    #         # Lightning will handle the gradient clipping
    #         self.clip_gradients(
    #             optimizer,
    #             gradient_clip_val=gradient_clip_val,
    #             gradient_clip_algorithm=gradient_clip_algorithm
    #         )
    #     # else:
            # implement your own custom logic to clip gradients for generator (optimizer_idx=0)

LOAD_LAYER_TYPES = (nn.ConvTranspose2d, ResBlock, nn.ReLU, nn.Conv2d)


def set_net_weights(source, target):
    for i, layer in enumerate(source):
        if hasattr(layer, 'weight'):
            target[i].weight = layer.weight
        if hasattr(layer, 'bias'):
            target[i].bias = layer.bias
        if isinstance(layer, ResBlock):
            set_net_weights(layer.conv, target[i].conv)
        if not isinstance(layer, LOAD_LAYER_TYPES):
            raise ValueError('Unexpected layer type, add support for new layers here')
        if not isinstance(target[i], LOAD_LAYER_TYPES):
            raise ValueError('Unexpected layer type, add support for new layers here')


def cli_main():
    torch.backends.cudnn.benchmark = True  # autotune kernels
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    # model args
    parser = LearnMax.add_reinforcement_learning_args(parser)
    parser = LearnMax.add_model_specific_args(parser)
    args, unknown = parser.parse_known_args()
    viz_dvq = args.viz_dvq or args.viz_all_dvq_clusters
    if args.num_workers is None:
        # data loader workers - pycharm has issues debugging when > 0
        # also weirdness when >0 in that wandb.init needs to be called for quantize to log???
        #   - must be due to spawning multiple training processes?
        args.num_workers = 0 if DEBUGGING else 0
        print('cli num workers', args.num_workers)
        print('DEBUGGING', DEBUGGING)

    if viz_dvq:
        args.training_gpt = False
        args.dvq_enable_kmeans = False
        args.warm_start_size = 100

    if DEBUGGING:
        wandb_name = None
        wandb_mode = 'disabled'
        fast_dev_run = False
        log.warning(f'Setting batch size to {args.gpt_batch_size}!')
    else:
        if viz_dvq:
            wandb_name = None
        else:
            wandb_name = input('\n\nExperiment name?\n\n')
        wandb_mode = 'online'
        fast_dev_run = False

    learn_max_args = copy(args.__dict__)
    del learn_max_args['num_gpus']  # This is for trainer later
    del learn_max_args['viz_dvq']  # Not for model
    del learn_max_args['viz_all_dvq_clusters']  # Not for model
    model = LearnMax(**learn_max_args)
    if args.dvq_checkpoint:
        load_pretrained_dvq(args, model)

    if args.viz_dvq is not None:
        return model.viz_dvq()
    elif args.viz_all_dvq_clusters:
        return model.viz_all_dvq_clusters()
    else:
        delattr(args, 'viz_dvq')
        delattr(args, 'viz_all_dvq_clusters')

    # common = {'batch_size': args.gpt_batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
    # trainer args  # TODO: Check that our defaults above are preserved for overlapping things like pin-memory
    parser.add_argument('-x', '--num_epochs', type=int, default=1000, help="number of epochs to train for")
    parser.add_argument('-p', '--pin_memory', type=bool, default=True, help="pin memory on dataloaders?")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    wandb.init(entity='crizcraig', save_code=True, name=wandb_name, mode=wandb_mode)
    # wandb.watch(model)  # causes OOM https://github.com/wandb/client/issues/2644

    wandb_logger = WandbLogger()
    log.info(json.dumps(vars(args), indent=0))
    """
    Algo
    Fill up replay buffer for a while, taking random actions to start. .populate()
    Train the dvq on the replay buffer randomly shuffled.
    Use dvq tokens to train transformer (gpt-architecture) 
    """

    seed_everything(SEED)  # env is seeded later tho

    if not args.dvq_checkpoint:
        train_dvq(args, model, wandb_logger, fast_dev_run)

    # return # TODO: GPT stuff below

    # -------------------- Standard mingpt training
    log.info("preparing the learning rate schedule")
    # number of tokens backpropped in one iteration, need to reduce for zuma vs char as tokens are 10x larger
    iter_tokens = args.gpt_batch_size * args.gpt_block_size
    epoch_tokens = math.ceil(args.batches_per_epoch * iter_tokens)
    lr_decay = GptWarmupCosineLearningRateDecay(learning_rate=6e-4,
                                                warmup_tokens=512 * 20,  # epoch_tokens // 2,

                                                final_tokens=args.num_epochs * epoch_tokens)
    t0 = time.time()
    log.info(f'training for {args.num_epochs} epochs...')
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', every_n_train_steps=1000, save_top_k=3,
                                          verbose=True)

    if os.getenv('CUDA_VISIBLE_DEVICES') == '':
        args.num_gpus = 0

    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=args.num_epochs,
                         # gradient_clip_val=1.0, # lightning does not support with manual optimization which we need due to two optimizers
                         callbacks=[lr_decay, checkpoint_callback],
                         precision=args.precision,
                         default_root_dir=args.default_root_dir,
                         logger=wandb_logger,
                         deterministic=True,  # Turn off deterministic to speed up
                         # overfit_batches=1,
                         fast_dev_run=fast_dev_run)

    # checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="max_interesting", mode="max", period=1, verbose=True)

    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=int(1.2e6), logger=wandb_logger,
    #                                         gpus=1)  # Max steps can be like 5k I think.

    trainer.fit(model)

    # train_dataset = CharDataset(open('train_shakespeare.txt', 'r').read(), args.block_size)  # one line of poem is roughly 50 characters
    # val_dataset = CharDataset(open('val_shakespeare.txt', 'r').read(), args.block_size)
    # test_dataset = CharDataset(open('test_shakespeare.txt', 'r').read(), args.block_size)
    #
    # common = {'batch_size': args.batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
    # train_dataloader = DataLoader(train_dataset, shuffle=True, **common)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, **common)

    t1 = time.time()
    log.info("%d epochs took %fs, or %fs/epoch" % (args.num_epochs, t1 - t0, (t1 - t0) / args.num_epochs))
    # end standard mingpt training


def load_pretrained_dvq(args, model):
    map_location = None
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        map_location = torch.device('cpu')
    _model = torch.load(args.dvq_checkpoint, map_location=map_location)
    state_dict = {k:v for k,v in _model['state_dict'].items() if k.startswith('dvq')}
    state_dict = {k[4:]:v for k,v in state_dict.items()}  # Remove `dvq.` as we're loading a submodule
    model.dvq.load_state_dict(state_dict)
    # _model = LearnMax.load_from_checkpoint(args.dvq_checkpoint)
    # set_net_weights(_model.dvq.encoder.net, model.dvq.encoder.net)
    # set_net_weights(_model.dvq.decoder.net, model.dvq.decoder.net)
    # assert len(list(model.dvq.decoder.net.parameters())) == 14, 'Did you add a new layer, if so copy the weights'
    # assert _model.dvq.quantizer.embed.embedding_dim == model.dvq.quantizer.embed.embedding_dim
    # assert _model.dvq.quantizer.embed.num_embeddings == model.dvq.quantizer.embed.num_embeddings
    # assert _model.dvq.quantizer.output_proj == model.dvq.quantizer.output_proj
    # assert _model.dvq.quantizer.patch_width == model.dvq.quantizer.patch_width
    # assert _model.dvq.is_single_token2 == model.dvq.is_single_token2
    # model.dvq.quantizer.embed.weight = _model.dvq.quantizer.embed.weight
    # model.dvq.quantizer.proj.weight = _model.dvq.quantizer.proj.weight
    # model.dvq.quantizer.proj.bias = _model.dvq.quantizer.proj.bias
    for attr in dir(model.dvq):
        if hasattr(getattr(model.dvq, attr), 'weight') and attr not in ('embed', 'proj'):
            raise ValueError('Unexpected params, add support for new layers here')


def train_dvq(args, model, wandb_logger, fast_dev_run):
    # -------------------- Standard dvq training
    # annealing schedules for lots of constants
    callbacks = [ModelCheckpoint(monitor='dvq_recon_loss', mode='min', every_n_train_steps=1000, save_top_k=3),
                 DecayLR()]
    if False and args.dvq_vq_flavor == 'gumbel':  # Not used yet
        callbacks.extend([DecayTemperature(), RampBeta()])
    # Things to try
    # More steps cos anneal goes to 1.2M! Not helping
    # Fewer clusters, we're only using ~50 out of 4096: Resulted in blurry images
    # Do k-means throughout training, not just once: works great! went from 10/20 to 19.5/20 correct images
    # Output image, we can count images that are obviously wrong
    # 10 points per cluster seemed to work better than 20, try 15, etc...: 15 works pretty well vs 20 and 10
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=int(1.2e6), logger=wandb_logger,
                                            gpus=args.num_gpus, fast_dev_run=fast_dev_run)  # Max steps can be like 5k I think.
    trainer.fit(model)


class ProcessFrame84Color(ObservationWrapper):
    """preprocessing images from env"""

    def __init__(self, env=None):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('This class uses OpenCV which it is not installed yet.')

        super(ProcessFrame84Color, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs"""
        return ProcessFrame84Color.process(obs)

    @staticmethod
    def process(frame):
        """image preprocessing, formats to 84x84"""
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        # x_t = np.reshape(x_t, [84, 84, 3])
        return x_t  # .astype(np.uint8)  # we convert to float32 in ScaledFloatFrame so just leave as float


def get_default_embeddings(is_single_token2=False):
    if is_single_token2:
        # length we project dvq vectors to for processing within the transformer
        # (large due to whole image reshaped into single token, no patches)
        default_embedding_dim = 4410
        # default_num_embeddings = 512
        default_num_embeddings = 4096
        # default_num_embeddings = 64: Works horribly on zuma
        # default_num_embeddings = 10 ** 4
    else:
        default_embedding_dim = 64
        default_num_embeddings = 512
    return default_num_embeddings, default_embedding_dim


if __name__ == '__main__':
    cli_main()
