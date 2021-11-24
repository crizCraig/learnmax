import argparse
import json
import math
import os
import time
from collections import namedtuple
from copy import copy
from typing import Tuple, List, OrderedDict, Dict, Optional

import cv2
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import wandb
import matplotlib.pyplot as plt
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

from learn_max import dvq
from learn_max.agent import LearnMaxAgent, AgentState
from learn_max.dvq.model.deepmind_enc_dec import ResBlock
from learn_max.utils import topk_interesting, _init_weights, get_batch_vars
from learn_max.constants import SAVE_DIR, SEED, DEBUGGING
from learn_max.dvq.vqvae import VQVAE, DecayLR, DecayTemperature, RampBeta
from learn_max.mingpt.lr_decay import GptWarmupCosineLearningRateDecay
from learn_max.mingpt.model import GPT


class LearnMax(pl.LightningModule):
    def __init__(
            self,
            # dvq_embedding_dim: int = 4410,  # length of embedding vectors output by dvq to transformers

            embedding_dim: int = None,
            num_embeddings: int = None,  # Number of possible discrete states shared between dvq and gpt

            # TODO: Add more levels of transformers for salient events

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
            gpt_learning_rate: float = 3e-4,  # the base learning rate of the model
            gpt_weight_decay: float = 0.1,  # amount of regularizing L2 weight decay on MatMul ops
            gpt_betas: Tuple[float, float] = (0.9, 0.95),  # momentum terms (betas) for the Adam optimizer
            gpt_embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
            gpt_resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
            gpt_attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix

            # Whether to use embeddings or indexes as input to first layer. With characters, we learn an embedding
            # from the ascii index, but with dvq inputs, the embeddings are high dimensional and
            # semantically meaningful so we use those as input.
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
        self.training_gpt = True if self.dvq_checkpoint is not None else False

        self.gpt_embedding_dim = embedding_dim  # the "width" of the model (embedding_dim), number of channels in each Transformer
        self.gpt_block_size = gpt_block_size
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
        self.n_actions = self.env.action_space.n

        # This action embedding will be concatenated with the dvq output. The first saliency level transformer
        #  will then enumerate all actions for all possible dvq tokens for n_actions * n_embd total possible
        #  outputs from the softmax. This allows selecting actions with the known next state.
        #  Saliency levels above that will have abstract actions and therefore can't be enumerated.
        #  Abstract actions will live in combined tokens along with abstract states. These tokens will be generated
        #  by dvq's which take in z,a's below them.
        self.action_emb = nn.Embedding(self.n_actions, embedding_dim)

        self.apply(_init_weights)  # Should only be action embeddings, other weights are in dvq and gpt

        # RL type tracking metrics
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.batch_states = []
        self.batch_actions = []

        self.state = self.reset()

        if self.is_single_token2:
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_embeddings, embedding_dim=self.dvq_embedding_dim,
                loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor, quantize_proj=dvq_quantize_proj,
                is_single_token2=self.is_single_token2, enable_kmeans=dvq_enable_kmeans)
        else:
            # TODO: Need to test that distance between clusters is further than avg distance between points in a cluster
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_embeddings, embedding_dim=self.dvq_embedding_dim,
                loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor, quantize_proj=None,
                is_single_token2=self.is_single_token2)

        self.gpt = GPT(vocab_size=num_embeddings, block_size=gpt_block_size, n_layer=gpt_n_layer,
                       embedding_dim=self.gpt_embedding_dim, n_head=gpt_n_head, learning_rate=gpt_learning_rate,
                       weight_decay=gpt_weight_decay, betas=gpt_betas, embd_pdrop=gpt_embd_pdrop,
                       resid_pdrop=gpt_resid_pdrop, attn_pdrop=gpt_attn_pdrop, should_input_embed=gpt_input_embed,
                       num_actions=self.n_actions)

        self.agent = LearnMaxAgent(model=self, num_actions=self.env.action_space.n)

    def reset(self):
        return self.env.reset(), AgentState()

    def run_n_episodes(self, env, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """
        Carries out N episodes of the environment with the current agent

        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action, agent_state = self.agent(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            self.state = self.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action, agent_state = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                new_state = (next_state, agent_state)
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=new_state)
                self.buffer.append(exp)
                self.state = new_state

                if done:
                    self.state = self.reset()

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

        while True:
            self.total_steps += 1
            action, agent_state = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            new_state = (next_state, agent_state)

            # TODO: Subtract 0.5 from image so that we - map [0,1] range to [-0.5, 0.5]

            # TODO: Allow learning within the episode

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=new_state)

            self.buffer.append(exp)
            # print(f'bufflen {len(self.buffer)}')
            self.state = new_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.reset()
                episode_steps = 0
                episode_reward = 0

            if self.training_gpt:
                # Sample sequentially from replay buffer
                # If training GPT, we need to return the dvq state indexes as well to be used in on_train_batch_end
                #   in lr_decay _, y = batch
                states, actions, rewards, dones, new_states, agent_states = self.sample_latest_sequential()
            else:
                # Sample randomly from replay buffer
                states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)  # TODO: Change to dvq_batch_size?
                # TODO: Test extracting agent_states
                states, _ = states
                new_states, agent_states = new_states

            for idx, _ in enumerate(dones):
                # Lightning wants tensors, numpy arrays, numbers, dicts or lists in batch
                agent_states_idx = [_.dict() for _ in agent_states[idx]]
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx], agent_states_idx

            # if self.total_steps % 1000 == 0:
            #     from guppy import hpy
            #     h = hpy()
            #     print('heap stats', h.heap())

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

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
        for _ in start_indices:
            # TODO: pad with -100 when last index? We just avoid by making sure we sample early enough in the buffer
            indices = np.array(list(range(len(self.buffer) - 1)[-self.gpt_block_size:]))
            states, actions, rewards, dones, next_states = zip(*[self.buffer.buffer[idx] for idx in indices])
            ret_states.append([s[0] for s in states])
            ret_actions.append(actions)
            ret_rewards.append(rewards)
            ret_dones.append(dones)
            ret_next_states.append([s[0] for s in next_states])
            ret_agent_states.append([s[1] for s in next_states])
        # TODO: Speed this up with numba?
        ret = (
            np.array(ret_states),
            np.array(ret_actions),
            np.array(ret_rewards, dtype=np.float32),
            np.array(ret_dones, dtype=bool),
            np.array(ret_next_states),
            ret_agent_states,
        )
        return ret

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> torch.Tensor:
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
            self.dvq.set_enable_kmeans(False)
            gpt_x, z_q_ind_x, z_q_ind_y = get_batch_vars(batch, return_agent_state=True, populate_gpt=True)

            gpt_loss = self.gpt.training_step(batch=(gpt_x, z_q_ind_x, z_q_ind_y), batch_idx=batch_idx)  # Train gpt on dvq tokens shifted for prediction

            # We use multiple optimizers so need to manually backward each one separately
            self.gpt_optimizer.zero_grad()
            # self.manual_backward(gpt_loss['loss'], self.gpt_optimizer)

            # Required due to https://github.com/PyTorchLightning/pytorch-lightning/issues/7698
            # Basically we run into issues with gradient clipping + manual_backward()
            self._running_manual_backward = True
            self.trainer.train_loop.backward(gpt_loss['loss'], optimizer=self.gpt_optimizer, opt_idx=None)
            self._running_manual_backward = False

            self.gpt_optimizer.step()

            loss = gpt_loss['loss']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss = loss.unsqueeze(0)
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

    def _train_step_dvq(self, batch, batch_idx):
        self.dvq.set_enable_kmeans(True)

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
            self.dvq.set_enable_kmeans(False)
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
        # TODO: Add the VQVAE and GPT args here as well
        arg_parser.add_argument("--dvq_quantize_proj", type=int, default=None)
        arg_parser.add_argument("--single_token2", action='store_true', default=False)
        arg_parser.add_argument('-n', '--num_workers', type=int, default=None, help="number of workers for dataloading")
        arg_parser.add_argument('--viz_dvq', type=str, help="visualize dvq images", default=None)
        arg_parser.add_argument('--dvq_checkpoint', type=str, help="Checkpoint to restore", default=None)
        arg_parser.add_argument('--gpt_batch_size', type=int, help="GPT batch size", default=4)
        arg_parser.add_argument('--gpt_block_size', type=int, default=80,
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

    def beam_search(self, z_q_emb, a_buff, beam_batch_size=64):
        """
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

        * take the transformer's input sequence of z,a embeddings and output deviations as input to beam search
          (note that deviation is the predicted output prob change and is the proxy we are using for uncertainty)
        * for saliency level zero, just concatenate state and action embeddings for input to the transformer,
          for higher levels, get the action that corresponds to each output state by forwarding it through the dvq decoder
          for that saliency level in batch(es) and getting the closest action embedding to the decoded one.
          saliency levels above level 0 will consist of their own dvq + gpt that clusters the action + state tokens
          from below and predicts those as abstract action-states. This is to allow for planning and creating abstract
          actions like, beat level 1 vs jump.
        * add deviations to _all_ encountered sorted deviations
            * try naive unoptimal way first though since we don't have that many branches)
            * If we need to optimize, use TORCH.SEARCHSORTED
                * insert new action_deviations with torch.cat - but also keep track of the associated sequence index in another
                  (unsorted) dimension of the the action_deviations pool
            * sequences just get appended to and are not sorted, so sequence index can be static.
        * get topk_interesting(action_deviations, k) output indexes - (nb. interesting is defined as within 50-75pct
          uncertainty) - however we may need to do 50-100pct to avoid state starvation
          there will be some trajectories at high saliency levels that are the most interesting. we hope that those
          trajectories remain most interesting for some amount of time so that we're not switching strats to often.
        * limit the initial actions in the trajectory to ones with a predicted state that's close to the one received from the sim
        * feed the corresponding z,a embeddings at those indexes back into the transformer at the end of the
          current sequence (we are simulating the env in multiple future branches)
        * get new states and deviations for the next level of the tree and add them to the sorted list according to above
        * when we build up a pool of transformer i/o that reaches some desired max size in GPU memory, get the most
          interesting trajectory and return the first action of that trajectory.
        * for the zeroth saliency level, just get the action part of the embedding, for higher levels, run the embedding through
          the dvq for that level to get the representation for the level below, and so on until level zero.
        * we can now throw away the search tree
        * add the previous state and action to the training batch for the transformer
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
        """

        # TODO:
        #   - Get action embeddings
        #   - Concatenate with z from dvq

        action_emb = self.action_emb(torch.tensor(a_buff)).squeeze()  # B, emb_dim = 128, 512

        deviations = self.gpt()

        beam_i = topk_interesting(deviations, k=beam_batch_size)
        # get actions associated with beam_i using decoder IN A BATCH
        # add these actions to appropriate sequences
        # add the new deviations for actions _after_ i.

        #    0.6  0.8  0.8  0.7
        pass


def viz_dvq():
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210819_112836-2sk8562w/files/learnmax-learn_max/2sk8562w/checkpoints/epoch=0-step=1179.ckpt')
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210821_213732-ujs1wtib/files/learnmax-learn_max/ujs1wtib/checkpoints/epoch=0-step=1299.ckpt')
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210822_114404-1hl1r5gh/files/learnmax-learn_max/1hl1r5gh/checkpoints/epoch=0-step=79.ckpt')
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210822_122545-1vpms2wc/files/learnmax-learn_max/1vpms2wc/checkpoints/epoch=0-step=999.ckpt')
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210823_122845-19bx2g56/files/learnmax-learn_max/19bx2g56/checkpoints/epoch=3-step=38899.ckpt')
    # model = LearnMax.load_from_checkpoint('/home/c2/src/learnmax/learn_max/wandb/run-20210824_140120-2w1glywq/files/learnmax-learn_max/2w1glywq/checkpoints/epoch=0-step=7399.ckpt')
    # ckpt = '/home/c2/src/learnmax/learn_max/wandb/run-20210906_120330-2y37nll3/files/learnmax-learn_max/2y37nll3/checkpoints/epoch=36-step=369999.ckpt'
    # ckpt = '/home/c2/src/learnmax/learn_max/wandb/run-20210907_154630-8v2mk188/files/learnmax-learn_max/8v2mk188/checkpoints/epoch=8-step=81999.ckpt'  # perfect on 20 examples
    ckpt = '/home/c2/src/learnmax/learn_max/wandb/run-20210917_122229-3h1loaoh/files/learnmax-learn_max/3h1loaoh/checkpoints/epoch=3-step=30999.ckpt'  # 19/20
    print(f'visualizing {ckpt}')
    model = LearnMax.load_from_checkpoint(ckpt)
    model.cuda()
    wandb.init(entity='crizcraig', mode='disabled')

    test_loader = model.test_dataloader()
    x = next(iter(test_loader))
    x = [t.cuda() for t in x]
    loss, recon_loss, latent_loss, x_hat = model.dvq.training_step(x, 0)
    # states, actions, rewards, dones, new_states = next(iter(test_loader))
    # x = torch.cat([states, new_states])
    xcols = torch.cat([x[0], x_hat[:32]], axis=2)  # side by side x_pre and xhat
    xrows = torch.cat([xcols[i] for i in range(x[0].size(0))], axis=2)

    plt.figure(figsize=(20, 5))
    plt.imshow((xrows.data.cpu().permute(1, 2, 0) + 0.5).clamp(0, 1))
    plt.axis('off')

    plt.show()

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
    args = parser.parse_args()
    if args.num_workers is None:
        # data loader workers - pycharm has issues debugging when > 0
        # also weirdness when >0 in that wandb.init needs to be called for quantize to log???
        #   - must be due to spawning multiple training processes?
        args.num_workers = 0 if DEBUGGING else 0
        print('cli num workers', args.num_workers)
        print('DEBUGGING', DEBUGGING)
    if args.viz_dvq is not None:
        return viz_dvq()
    else:
        delattr(args, 'viz_dvq')

    model = LearnMax(**args.__dict__)
    if args.dvq_checkpoint:
        load_pretrained_dvq(args, model)
        args.enable_kmeans = False

    # common = {'batch_size': args.gpt_batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
    # trainer args  # TODO: Check that our defaults above are preserved for overlapping things like pin-memory
    parser.add_argument('-x', '--num_epochs', type=int, default=2, help="number of epochs to train for")
    parser.add_argument('-g', '--num_gpus', type=int, default=1, help="number of gpus to train on")
    parser.add_argument('-p', '--pin_memory', type=bool, default=True, help="pin memory on dataloaders?")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # TODO: Try comet for code saving, it looks like it saves and does diffs on everything, not just this file
    if DEBUGGING:
        wandb_name = None
        wandb_mode = 'disabled'
        fast_dev_run = True
        if os.environ.get('DEBUG_GPU', 'n') == 'n':
            args.num_gpus = 0
    else:
        wandb_name = input('\n\nExperiment name?\n\n')
        wandb_mode = 'online'
        fast_dev_run = False
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
    log.info("training...")
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min', every_n_train_steps=1000, save_top_k=3,
                                          verbose=True)
    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0,
                         callbacks=[lr_decay, checkpoint_callback],
                         precision=args.precision, default_root_dir=args.default_root_dir, logger=wandb_logger,
                         deterministic=True, fast_dev_run=fast_dev_run)  # Turn off deterministic speed up

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
    _model = torch.load(args.dvq_checkpoint)
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
