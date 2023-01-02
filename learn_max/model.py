import json
import math
import os
import sys
import time
from collections import deque, defaultdict
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
# from pl_bolts.datamodules.experience_source import Experience
from pl_bolts.models.rl.common.gym_wrappers import (
    make_environment, ImageToPyTorch, FireResetEnv, ScaledFloatFrame)
from pl_bolts.utils import _OPENCV_AVAILABLE
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn import functional as F

from learn_max.config import get_train_args_from_cli, get_model_args_from_cli
from learn_max.agent import LearnMaxAgent, AgentState
from learn_max.dvq.model.deepmind_enc_dec import ResBlock
from learn_max.mingpt.utils import get_num_output_embeddings
from learn_max.replay_buffer import ReplayBuffers, Experience
from learn_max.salience.detect_salience import detect_salience
from learn_max.salience.salience import SalienceLevel, salience_level_factory
from learn_max.utils import topk_interesting, _init_weights, get_date_str, get_action_states, \
    wandb_log, no_train, wandb_check_flush, best_effort, viz_experiences, viz_salience
from learn_max.constants import (SAVE_DIR, SEED, DEBUGGING, DATE_STR,
                                 ROOT_DIR, RUN_ID, WANDB_MAX_LOG_PERIOD,
                                 ACC_LOG_PERIOD, MAX_NUM_SALIENCE_LEVELS)
from learn_max.dvq.vqvae import VQVAE, DecayLR, DecayTemperature, RampBeta
from learn_max.mingpt.lr_decay import GptWarmupCosineLearningRateDecay
from learn_max.mingpt.model import GPT
from learn_max.viz_predicted_trajectories import get_action_text, get_np_txt_caption2

log.remove()
log.add(sys.stderr, level="INFO")
log.add('learnmax_log_{time}.log', compression='zip', rotation='10 MB')

class LearnMax(pl.LightningModule):
    def __init__(
            self,
            checkpoint: str = None,  # Model checkpoint

            embedding_dim: int = None,
            num_state_embeddings: int = None,  # Number of possible discrete states shared between dvq and gpt

            # dvq args - dvq = deep vector quantization
            dvq_batch_size: int = 32,  # DVQ batch size
            dvq_n_hid: int = 64,  # number of channels controlling the size of the model
            # dvq_embedding_dim: int = 4410,  # now embedding_dim but may separate from GPT embedding dim) length of embedding vectors output by dvq to transformers
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
            gpt_seq_len: int = 80,
            gpt_n_layer: int = 8,  # depth of the model; number of Transformer blocks in sequence
            gpt_n_head: int = 10,  # number of heads in each multi-head attention inside each Transformer block
            gpt_batch_size: int = 16,  # with such large embeddings (4410) needs to be small now to fit on rtx 2080

            # mingpt model optimization args
            should_train_gpt: bool = False,  # whether to train gpt
            gpt_learning_rate: float = 3e-4,  # the base learning rate of the model (overwritten by GptWarmupCosineLearningRateDecay)
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
            gamma: float = 1,  # discount factor - only used for evaluation metrics right now
            n_steps: int = 1,  # number of steps to return from each environment at once
            env_id: str = 'MontezumaRevenge-v0',  # gym environment tag
            warm_start_size: int = 10_000,  # how many samples do we use to fill our buffer at the start of training
            batches_per_epoch: int = 10_000,  # number of batches per pseudo (RL) epoch
            actions_per_batch: int = 1,  # adds more samples per action to prevent overfitting

            # Tree search
            beam_width: int = 3,  # How many trajectories to predict in tree search
            num_search_steps: int = 10,  # How many steps to predict out in each trajectory

            # Standard stuff
            num_workers: int = 0,  # data loader workers - pycharm has issues debugging these. also gen batch requires NN for action so can't use > 0 at runtime either yet
            data_dir: str = SAVE_DIR,  # place to save tfboard logs and checkpoints
            batch_size: int = 32,  # do we have a batch size? or are gpt and dvq batch sizes adequate?
            # checkpoint: str = None, # Checkpoint to restore from

            single_token2: bool = True,

            should_viz_predict_trajectory: bool = False,
            should_overfit_gpt: bool = False,  # Whether to overfit on a small batch of recent experience
            train_to_test_collection_ratio: float = 10,  # Number of train to test files to collect in replay buffer
            # Dataloader
            pin_memory: bool = False,

            # Salience
            use_internal_salience: bool = True,  # whether to use internal signals for salience
            salience_resume_path: str = None,  # path to salience pickles dir
            salience_use_logits: bool = False,  # whether to use logits in salience
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

        if num_state_embeddings is None:
            num_state_embeddings, embedding_dim = default_num_embeddings, default_embedding_dim

        # Hyperparameters
        self.dvq_embedding_dim = embedding_dim  # size of the embedding vector representing a cluster of embeddings
        self.dvq_n_hid = dvq_n_hid
        self.dvq_num_embeddings = num_state_embeddings
        self.dvq_loss_flavor = dvq_loss_flavor
        self.dvq_input_channels = dvq_input_channels
        self.dvq_enc_dec_flavor = dvq_enc_dec_flavor
        self.dvq_vq_flavor = dvq_vq_flavor
        self.dvq_checkpoint = dvq_checkpoint
        self.dvq_ready = False  #  Whether the dvq has enough data to train a batch
        self.dvq_batch_size = dvq_batch_size

        self.should_train_gpt = should_train_gpt
        self.should_overfit_gpt = should_overfit_gpt

        # the "width" of the model (embedding_dim), number of channels in each Transformer
        self.gpt_input_embedding_dim = embedding_dim
        self.gpt_seq_len = gpt_seq_len  # Size of the temporal window
        self.gpt_batch_size = gpt_batch_size
        self.gpt_n_layer = gpt_n_layer
        self.gpt_n_head = gpt_n_head
        self.gpt_learning_rate = gpt_learning_rate
        self.gpt_weight_decay = gpt_weight_decay
        self.gpt_betas = gpt_betas
        self.gpt_embd_pdrop = gpt_embd_pdrop
        self.gpt_resid_pdrop = gpt_resid_pdrop
        self.gpt_attn_pdrop = gpt_attn_pdrop

        # RL / sensorimotor stuff
        self.avg_reward_len = avg_reward_len
        self.gamma = gamma
        self.n_steps = n_steps
        self.env_id = env_id
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.actions_per_batch = actions_per_batch
        self.data_dir = data_dir
        self.total_steps = 0
        self.total_training_steps = 0
        self.total_rewards = []
        self.total_episode_steps = []

        # Tree search
        self.beam_width = beam_width
        self.num_search_steps = num_search_steps


        # Visualization
        self.should_viz_predict_trajectory = should_viz_predict_trajectory

        # Salience
        self.use_internal_salience = use_internal_salience
        self.internal_salience_update_freq = self.gpt_seq_len // 2
        self.salience_resume_path = salience_resume_path
        self.salience_use_logits = salience_use_logits

        self.pin_memory = pin_memory
        self.num_workers = num_workers

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


        """
        Zuma action meanings
        0 Action.NOOP
        1 Action.FIRE - jump
        2 Action.UP - move up ladder
        3 Action.RIGHT - walk right
        4 Action.LEFT - walk left
        5 Action.DOWN - down ladder
        6 Action.UPRIGHT - move up right ladder not needed
        7 Action.UPLEFT - move up left ladder not needed
        8 Action.DOWNRIGHT - move down right ladder not needed
        9 Action.DOWNLEFT - move up left ladder not needed
        10 Action.UPFIRE - jump up ladder not needed
        11 Action.RIGHTFIRE - jump right ladder not needed
        12 Action.LEFTFIRE - jump left ladder not needed
        13 Action.DOWNFIRE - jump down ladder not needed
        14 Action.UPRIGHTFIRE - jump up right ladder not needed
        15 Action.UPLEFTFIRE - jump up left ladder not needed
        16 Action.DOWNRIGHTFIRE - jump down right ladder not needed
        17 Action.DOWNLEFTFIRE - jump down left ladder not needed
        """
        log.warning('Reducing action set for Montezuma\'s revenge')
        # Reduce action space to speed up training iteration, should converge to same performance without this
        self.env.unwrapped._action_set = [
            self.env.unwrapped._action_set[0],  # noop
            self.env.unwrapped._action_set[1],  # fire (jump in zuma)
            self.env.unwrapped._action_set[2],  # up (on ladder in zuma)
            self.env.unwrapped._action_set[3],  # right
            self.env.unwrapped._action_set[4],  # left
            self.env.unwrapped._action_set[5],  # down  (on ladder in zuma)
        ]
        self.env._action_space = gym.spaces.Discrete(len(self.env.unwrapped._action_set ))
        self.num_actions = self.env.action_space.n  # TODO: Change for zuma limited actions

        log.info(
            'Action map\n' +
            '\n'.join(f'{i} {a}' for i, a in enumerate(self.env.unwrapped._action_set))  # TODO: Change for zuma limited actions
        )

        self.obs_shape = self.env.observation_space.shape


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
        self.episode_steps = 0
        self.episode_reward = 0

        # Replay buffers (test and train)
        short_term_mem_max_length = (
            1000 if self.should_overfit_gpt else self.gpt_seq_len * self.gpt_batch_size
        )
        self.short_term_mem_max_length = short_term_mem_max_length
        self.gpt_frames_per_file = (
                self.gpt_seq_len * self.gpt_batch_size +
                self.get_num_extra_seq_samples()
        )
        self.should_overfit_gpt = should_overfit_gpt
        self.train_to_test_collection_ratio = train_to_test_collection_ratio
        self.replay_buffers = ReplayBuffers(
            env_id=self.env_id,
            short_term_mem_max_length=short_term_mem_max_length,
            overfit_to_short_term=should_overfit_gpt,
            train_to_test_collection_ratio=train_to_test_collection_ratio,
            # Get a full val batch when training starts
            frames_per_file=max(self.dvq_batch_size, self.gpt_frames_per_file),
            salience_level=0,
        )

        self.train_buf = self.replay_buffers.train_buf
        self.test_buf = self.replay_buffers.test_buf
        self.recent_experience = self.replay_buffers.short_term_mem
        self.predicted_trajectories_size = self.recent_experience.maxlen

        # Trajectory visualization data
        self.predicted_trajectory_buffer = deque(maxlen=self.predicted_trajectories_size if 'DEBUG_TRAJ_BUFF' not in os.environ else 10)

        if self.is_single_token2:
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_state_embeddings, embedding_dim=self.dvq_embedding_dim,
                             loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                             enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor,
                             quantize_proj=dvq_quantize_proj,
                             is_single_token2=self.is_single_token2, enable_kmeans=dvq_enable_kmeans)
            num_output_embeddings = num_state_embeddings * self.num_actions
            num_input_embeddings = num_output_embeddings
            tokens_in_frame = 1
        else:
            # TODO: Need to test that distance between clusters is further than avg distance between points in a cluster
            self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=num_state_embeddings, embedding_dim=self.dvq_embedding_dim,
                             loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                             enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor, quantize_proj=None,
                             is_single_token2=self.is_single_token2)
            num_output_embeddings = get_num_output_embeddings(
                num_state_embeddings,
                self.num_actions,
            )
            state_tokens_in_frame = self.dvq.encoder.out_width ** 2  # H*W patches
            tokens_in_frame = state_tokens_in_frame + 3  # 1 action + 1 delimiter + 1 salience

            # Since we don't need to predict salience (it's just the same as the input)
            #  we only add it to the input vocab cardinality, not the output / softmax size
            num_input_embeddings = num_output_embeddings + MAX_NUM_SALIENCE_LEVELS

        self.gpt = GPT(
            output_size=num_output_embeddings,
            num_input_embeddings=num_input_embeddings,
            num_state_embeddings=num_state_embeddings,
            frames_in_sequence_window=gpt_seq_len,
            n_layer=gpt_n_layer,
            input_embedding_dim=self.gpt_input_embedding_dim,
            n_head=gpt_n_head,
            learning_rate=gpt_learning_rate,
            weight_decay=gpt_weight_decay,
            betas=gpt_betas,
            embd_pdrop=gpt_embd_pdrop,
            resid_pdrop=gpt_resid_pdrop,
            attn_pdrop=gpt_attn_pdrop,
            should_input_embed=gpt_input_embed,
            num_actions=self.num_actions,
            is_single_token2=single_token2,
            state_tokens_in_frame=state_tokens_in_frame,
            tokens_in_frame=tokens_in_frame,
            batch_size=self.gpt_batch_size,
        )

        self.salience_levels = [self.get_salience_level(salience_resume_path, level=1)]
        self.agent = LearnMaxAgent(model=self, num_actions=self.num_actions)
        self.reset()

    def get_salience_level(self, salience_resume_path, level):
        return salience_level_factory(
            level=level,
            env_id=self.env_id,
            short_term_mem_max_length=self.short_term_mem_max_length,
            frames_per_file=self.gpt_frames_per_file,  # TODO: Increase this once 1 is working
            frames_in_sequence_window=self.gpt.frames_in_sequence_window,
            should_overfit_gpt=self.should_overfit_gpt,
            train_to_test_collection_ratio=self.train_to_test_collection_ratio,
            salience_resume_path=salience_resume_path,
        )

    def reset(self):
        self.agent_state = self.get_agent_state(self.env.reset())
        return self.agent_state

    def get_agent_state(self, state):
        """
        Populate agent state with external and internal (dvq only) state.

        We avoid storing GPT logits as we expect those to change more frequently when the network updates
        and want salience detection to get those fresh logits with forward.
        """
        if not isinstance(state, list):
            state = [state]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state), device=self.device)  # Cause issues when num_workers > 0

        # gpt_logits = None
        if self.should_train_gpt:
            dvq_loss, latent_loss, recon_loss, x, x_hat, z_q_emb, z_q_flat, z_q_ind = self.get_dvq_outputs(state)
            # if action is not None and not self.is_single_token2:
            #     pass
            #     with torch.no_grad():
            #         gpt_logits, _ = self.gpt.forward(z_q_ind.reshape(1, -1), torch.tensor([action], device=self.device))
        else:
            dvq_loss, latent_loss, recon_loss, x, x_hat, z_q_emb, z_q_flat, z_q_ind = (
                None, None, None, None, None, None, None, None
            )
        # Allow GPT to forward and store intermediate activations used for backprop
        # TODO: Check if we need to do requires_grad_() - I think it's already being done for us
        # z_q_flat.detach_()

        agent_state = AgentState(state=state,
                                 dvq_x=x,
                                 dvq_x_hat=x_hat,
                                 dvq_z_q_emb=z_q_emb,
                                 dvq_z_q_flat=z_q_flat,
                                 dvq_latent_loss=latent_loss,
                                 dvq_recon_loss=recon_loss,
                                 dvq_loss=dvq_loss,
                                 dvq_z_q_ind=z_q_ind,
                                 # gpt_logits=gpt_logits,
                                 )
        return agent_state

    @no_train
    def get_dvq_outputs(self, state):
        # TODO: Check if we should train DVQ and if not, put this in a no_train block
        x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.dvq.forward(state)
        return dvq_loss, latent_loss, recon_loss, x, x_hat, z_q_emb, z_q_flat, z_q_ind

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

    @no_train
    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            self.reset()
            while len(self.train_buf) < warm_start:
                self.step()

    def step(self):
        action, predicted_trajectory = self.agent.get_action(self.agent_state, self.device)
        next_state, reward, done, info = self.env.step(action[0])
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += reward
        next_agent_state = self.get_agent_state(next_state)
        exp = Experience(state=self.agent_state, action=action[0], reward=reward, done=done,
                         new_state=next_agent_state)

        self.replay_buffers.append(exp)

        self.predicted_trajectory_buffer.append(predicted_trajectory)

        if (
                self.use_internal_salience and
                self.replay_buffers.is_train() and
                self.total_steps % 2 * self.internal_salience_update_freq == 0  # TODO: parameterize this 2? It's the minimum for sequence diff so...
        ):
            # Use internal signals (e.g. logits) for salience
            # start_detect = time.time()
            self.detect_salience()
            # log.info(f'Salience detection took {time.time() - start_detect} seconds')

        # Creates new AgentState from same next_state as above so replay_i is correct
        self.agent_state = self.get_agent_state(next_state)

        if done:
            wandb_log({'episode_steps': self.episode_steps})
            self.done_episodes += 1
            self.total_rewards.append(self.episode_reward)
            self.total_episode_steps.append(self.episode_steps)
            self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
            self.reset_after_death()
            self.episode_steps = 0
            self.episode_reward = 0

    @no_train
    def detect_salience(self):
        assert len(self.replay_buffers.short_term_mem) >= self.gpt.frames_in_salient_event, \
            'Not enough contiguous experience to detect salience'
        recent_exp = self.get_recent_experience(
            max_size=self.gpt.frames_in_salient_event, train_buff_only=True)
        (actions,
         just_died,
         look_back,
         z_q_embed,
         z_q_ind,
         replay_ind,
         ) = recent_exp
        # TODO: Try feeding in longer sequences? We'll get more efficient use of windows in detect_salience, but
        #   have more potential overlap between steps when weights may not have changed much.

        if look_back >= self.gpt.frames_in_salient_event:
            # We have 2 sequences to use for detecting salience
            B, S, H, W = z_q_ind.shape  # batch, frames in sequence, height, width

            if self.salience_use_logits:
                use_emb = False
                gpt_logits = self.gpt.get_salience_logits(actions, z_q_ind, z_q_embed,
                                                          replay_ind)
                if gpt_logits is None:
                    return
            else:
                use_emb = True
                gpt_logits = None

            # This is the only place we should be calling detect_salience
            # TODO: Detect salience at higher levels
            saliences = detect_salience(
                actions=actions,
                z_q_ind=z_q_ind,
                z_q_emb=z_q_embed,
                replay_ind=replay_ind,
                seq_len=self.gpt_seq_len,
                frames_in_sequence_window=self.gpt.frames_in_sequence_window,
                tokens_in_frame=self.gpt.tokens_in_frame,
                state_tokens_in_frame=self.gpt.state_tokens_in_frame,
                num_state_embeddings=self.gpt.num_state_embeddings,
                num_actions=self.num_actions,
                tdigest=self.salience_levels[0].tdigest,
                use_emb=use_emb,
                logits=gpt_logits,
            )
            salience_ratio = len(saliences) / len(actions)
            wandb_log({'salience/ratio_all': salience_ratio})

            if saliences:
                salience_add = self.salience_levels[0].add(
                    saliences,
                    self.dvq.decoder,
                    self.device,
                    FiS=self.gpt.frames_in_sequence_window,
                    lower_replay_buffers=self.replay_buffers
                )
                if len(salience_add['repeats']) > 0:
                    ratio_repeats = len(salience_add['repeats']) / len(actions)
                    log.info(
                        f'Found {len(salience_add["repeats"])} already seen salient event(s) out of' +
                        f' {len(actions)} salient events (s) = {ratio_repeats * 100:.2f}%'
                    )
                if salience_add['new_salience_level']:
                    if len(self.salience_levels) == 1:
                        log.warning('Only two salience levels supported, skipping new cluster')
                    else:
                        log.info(f'New salience level {len(self.salience_levels)}')
                        self.salience_levels.append(
                            self.get_salience_level(None, level=len(self.salience_levels) + 1)
                        )
                        # TODO: Handle recursive detection of higher salience levels

                salient_replay_ind = [s['replay_ind'] for s in saliences]
                FiS = self.gpt.frames_in_sequence_window
                if salient_replay_ind:
                    viz_salience(FiS, salient_replay_ind, self.replay_buffers,
                                 self.dvq.decoder, self.device)

    def training_batch_gen(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields tuple containing  state, action, reward, done, next_state, and agent_state dict.
        """

        episode_reward = 0
        episode_steps = 0

        # self.profile_cuda_usage_tensors()

        (
            actions,
            agent_states,
            dones,
            new_states,
            rewards,
            states,
            indices,
            salience_levels,
        ) = (None, None, None, None, None, None, None, None)

        while True:
            if self.total_training_steps == 0 or 'OVERFIT' not in os.environ:
                # TODO: Move this tuple into a @dataclass
                (
                    states,
                    actions,
                    rewards,
                    dones,
                    new_states,
                    agent_states,
                    next_agent_states,
                    indices,
                    salience_levels,
                ) = self._take_action_and_sample_batch(episode_reward, episode_steps)
                # TODO: Add salience context token to all transformer training
                # TODO: Create embedding for goal token and concatenate with other tokens
                # TODO: Determine if we have enough salient events to train on at each level
                #   Prioritize higher levels in the batch. Fill batch with lower levels if needed.
                # TODO: Go through self.salience_levels and fill batch with top level exp first
                # TODO: Once we overfit to high level salience, try to sample randomly from all
                #   levels:
                #   https://excalidraw.com/#json=xEaCRq4tsmJiRCeoPPdCV,OiyAEfgHyNf2kYZ1S9dMXA
            if dones is None:
                log.error('dones is None, what is going on??? - trying to continue')
                time.sleep(5)
                continue
            for idx, _ in enumerate(dones):
                # Lightning wants tensors, numpy arrays, numbers, dicts or lists in batch
                agent_state = self.agent_states_to_dict(agent_states[idx])
                yield (
                    states[idx],
                    actions[idx],
                    rewards[idx],
                    dones[idx],
                    new_states[idx],
                    agent_state,
                    salience_levels[idx],
                )

            self.total_training_steps += 1

            # if self.total_steps % 1000 == 0:
            #     from guppy import hpy
            #     h = hpy()
            #     print('heap stats', h.heap())

            # Simulates epochs
            if self.total_training_steps % self.batches_per_epoch == 0:
                break

    def validation_batch_gen(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.should_train_gpt:
            log.warning('Validation not implemented for dvq')
            return
        if len(self.replay_buffers.test_buf) == 0:
            yield from self._yield_empty_validation_batch()
        else:
            assert len(self.test_buf) >= self.gpt_batch_size * self.gpt_seq_len
            start_indices = np.arange(start=0,
                                      stop=self.gpt_batch_size * self.gpt_seq_len,  # stop is non-inclusive
                                      step=self.gpt_seq_len)
            if len(start_indices) == 0:
                yield from self._yield_empty_validation_batch()
            else:
                assert len(start_indices) >= self.gpt_batch_size
                # TODO: Move this tuple into a @dataclass
                # TODO: Allow non-sequential (random) sequence (not frame) sampling for more IID validation
                ret = self.sample_sequential(
                    buffer=self.test_buf,
                    start_indices=start_indices,
                    seq_len=self.gpt_seq_len,
                )
                # TODO: Add salience level in sample_from_replay_buf() and remove in _get_gpt_train_batch
                if ret is None:
                    yield from self._yield_empty_validation_batch()
                else:
                    (states, actions, rewards, dones, new_states, agent_states,
                     next_agent_states, salience_levels) = ret
                    for idx, _ in enumerate(dones):
                        # Lightning wants tensors, numpy arrays, numbers, dicts or lists in batch
                        agent_state = self.agent_states_to_dict(agent_states[idx])
                        yield (states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx],
                               agent_state, salience_levels)

    @staticmethod
    def _yield_empty_validation_batch():
        log.warning('Generating empty validation data! Val stats will be wrong. '
                    'Make sure you fill test_buf with at least one gpt block before calling')

        yield (torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
               torch.tensor(0), torch.tensor(0), torch.tensor(0))

    def agent_states_to_dict(self, thing):
        if isinstance(thing, AgentState):
            return thing.dict()
        if isinstance(thing[0], AgentState):
            # agents_states only has a batch dimension, not window size
            return [_.dict() for _ in thing]

    def _take_action_and_sample_batch(self, episode_reward, episode_steps):
        for a_i in range(self.actions_per_batch):
            # TODO: If there's a very surprising experience, train on it right away
            self.step()

        if self.should_train_gpt:
            ret = self._get_gpt_train_batch()
        else:
            ret = self._get_dvq_train_batch()
        return ret

    def _get_gpt_train_batch(self):
        """
        Returns:
            states,
            actions,
            rewards,
            dones,
            new_states,
            agent_states,
            next_agent_states,
            replay_ind,
            salience_levels,
        """
        if len(self.train_buf) < (self.gpt_seq_len * self.gpt_batch_size):
            log.info('Train buff not full enough to run a batch yet, skipping')
            return None, None, None, [], None, None, None, None
        # Sample sequentially from replay buffer
        # If training GPT, we need to return the dvq state indexes as well to be used in on_train_batch_end
        #   in lr_decay _, y = batch
        # Sample 2 extra as we shift actions back 1 for action-states and targets forward 1 see sa2as()
        num_extra_seq_items = self.get_num_extra_seq_samples()
        seq_len = self.gpt_seq_len + num_extra_seq_items  # frames in sequence
        sample_size = self.gpt_batch_size * seq_len
        items_remaining = sample_size
        samples = []
        for level in self.salience_levels:
            buf = level.replay_buffers.train_buf
            level_sample_size = min(items_remaining, len(buf))
            if level_sample_size >= self.gpt_seq_len:
                sample = self.sample_from_replay_buf(
                    buf,
                    seq_len,
                    level_sample_size
                )
                if sample is not None:
                    # if this contains full sequences

                    # TODO: Add salience levels in sample_from_replay_buf() so validation gets it too
                    sample.append([[level] * len(sample[0])])

                    assert len(sample[1].shape) == 2  # action shape should be (batch, seq_len)
                    items_remaining -= np.prod(sample[1].shape)  # use num actions as proxy for num frames
                    samples.append(sample)

        sensor_sample = self.sample_from_replay_buf(
            self.replay_buffers.train_buf,
            seq_len,
            items_remaining,
        )

        # Salience level 0 should always have >= 1 full sequence
        assert sensor_sample is not None
        sensor_sample += [[0] * len(sensor_sample[0])]  # level = 0
        samples.append(sensor_sample)

        assert len(samples[-1][1].shape) == 2  # action shape should be (batch, seq_len)
        items_remaining -= np.prod(samples[-1][1].shape)  # use num actions as proxy for num frames
        assert items_remaining == 0

        states = torch.cat(tuple(r[0] for r in samples))
        actions = np.concatenate(tuple(r[1] for r in samples))
        rewards = np.concatenate(tuple(r[2] for r in samples))
        dones = np.concatenate(tuple(r[3] for r in samples))
        new_states = torch.cat(tuple(r[4] for r in samples))
        agent_states = tuple(np.concatenate(tuple(r[5] for r in samples)))
        next_agent_states = tuple(np.concatenate(tuple(r[6] for r in samples)))
        replay_ind = np.concatenate(tuple(r[7] for r in samples))
        salience_levels = tuple(np.concatenate(tuple(r[8] for r in samples)))

        return (
            states,
            actions,
            rewards,
            dones,
            new_states,
            agent_states,
            next_agent_states,
            replay_ind,
            salience_levels,
        )

    def _get_dvq_train_batch(self):
        # Sample single states randomly from replay buffer
        # agent_states, actions, rewards, dones, new_agent_states = self.buffer.sample(
        #     self.dvq_batch_size)
        replay_ind = np.random.choice(len(self.train_buf), self.dvq_batch_size, replace=False)
        exps = tuple(self.train_buf.get(idx)[0] for idx in replay_ind)
        states, actions, rewards, dones, new_states, agent_states, next_agent_states = (
            torch.cat([exp.state.state for exp in exps]),
            np.array([exp.action for exp in exps]),
            np.array([exp.reward for exp in exps], dtype=np.float32),
            np.array([exp.done for exp in exps], dtype=bool),
            torch.cat([exp.new_state.state for exp in exps]),
            [exp.state for exp in exps],
            [exp.new_state for exp in exps],
        )
        ret = (
            states,
            actions,
            rewards,
            dones,
            new_states,
            agent_states,
            next_agent_states,
            replay_ind,
        )
        return ret

    def sample_from_replay_buf(self, replay_buf, seq_len, sample_size):
        start_range = np.arange(len(replay_buf) - sample_size)
        # TODO: Allow non-sequential (random) sequence sampling for more IID training.
        #  In fact, we don't need sequential sampling here at all anymore as salience
        #  detection is now done in step by sampling from short term mem.

        # Sample sequential start indices
        # TODO: Just do one get() from replay buffer with entire sequential batch
        start_idx = np.random.choice(start_range, 1, replace=False)[0]
        start_indices = np.arange(start=start_idx,
                                  stop=start_idx + sample_size - seq_len,
                                  step=self.gpt_seq_len)
        if 'ALWAYS_SAMPLE_LATEST' in os.environ:
            start_indices[0] = start_range[-1]

        samples = self.sample_sequential(
            buffer=replay_buf,
            start_indices=start_indices,
            seq_len=seq_len,
        )
        if samples is None:
            return None

        (states, actions, rewards, dones, new_states, agent_states,
         next_agent_states) = samples
        replay_ind = start_indices
        return [
            states,
            actions,
            rewards,
            dones,
            new_states,
            agent_states,
            next_agent_states,
            replay_ind,
        ]

    def get_num_extra_seq_samples(self):
        if self.is_single_token2:
            num_extra_seq_items = 2  # action => states and shift targets by 1
        else:
            # Need a whole extra image to shift targets by one patch :/
            num_extra_seq_items = 1
        return num_extra_seq_items

    def reset_after_death(self):
        # TODO: The replay buffer should know reset boundaries so that we can avoid sampling across resets.
        #   This is important for AGI safety in that humans / life cannot respawn, and AI should reason about
        #   exploration in the same context.
        #   We sample in both the DVQ with sample(), but also in tree search by directly accessing the buffer (DONE)
        #   We also sample in train GPT batch in sample_sequential with random start points and need to
        #   NOT sample across resets here.
        # TODO: We need to add a death prediction logistic regression head to GPT for tree search to use to reduce
        #   expected entropy/reward for post-death actions by prob(death)
        # If in tree search and just died, then is_done=True, return early, no need to search
        dead_state = self.agent_state
        spawn_state = self.reset()
        self.replay_buffers.append(Experience(
            state=dead_state,
            # Reset action is noop - refer to self.env.unwrapped._action_set  # TODO: Use limited zuma action set
            # to ensure this is true for your env. Montezuma=Yes
            action=0,
            reward=0,
            done=False,
            new_state=spawn_state))
        self.predicted_trajectory_buffer.append(None)

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

    def sample_sequential(self, buffer, start_indices, seq_len):
        """Get batch size X gpt_seq_len windows for GPT"""
        ret_states = []
        ret_agent_states = []
        ret_actions = []
        ret_rewards = []
        ret_dones = []
        ret_next_states = []
        ret_next_agent_states = []
        for start in start_indices:
            # We don't pad with -100 for last index as we make sure to
            # sample early enough in the buffer to get full sequences
            exps = buffer.get(start, seq_len, device=self.device)
            if len(exps) < seq_len:
                return None
            agent_states, actions, rewards, dones, next_agent_states, replay_indexes, splits = zip(*exps)
            ret_states.append([ags.state for ags in agent_states])
            ret_agent_states.append(agent_states)
            ret_actions.append(actions)
            ret_rewards.append(rewards)
            ret_dones.append(dones)
            ret_next_states.append([ags.state for ags in next_agent_states])
            ret_next_agent_states.append(next_agent_states)
        # TODO: Speed this up with numba?
        # TODO: Just put everything in the agent_states and next_agent_states dict's

        ret = (
            torch.stack([torch.cat(rs) for rs in ret_states]),
            np.array(ret_actions),
            np.array(ret_rewards, dtype=np.float32),
            np.array(ret_dones, dtype=bool),
            torch.stack([torch.cat(rs) for rs in ret_next_states]),
            ret_agent_states,
            ret_next_agent_states,
        )
        return ret

    def get_compressed_cluster(self, z_q_ind):
        return self.compressed_dict[z_q_ind]

    def get_state_index(self, z_q_ind):
        if self.use_compressed_clusters:
            return self.get_compressed_cluster(z_q_ind)
        else:
            return z_q_ind

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

        if self.should_train_gpt:
            self.dvq.set_do_kmeans(False)
            if self.gpt_learning_rate == 0 or 'ZERO_LR' in os.environ:
                # For testing that train step and visualize methods produce same output, basically disable dropout
                self.eval()
            if self.is_single_token2:
                gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y, s, agent_state = self.gpt.get_batch_vars(batch)
                gpt_ret = self.gpt.training_step(batch=(gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y), batch_idx=batch_idx)
            else:
                gpt_ind, actions, salience_levels = self.gpt.get_batch_vars(batch)
                gpt_ret = self.gpt.training_step(
                    batch=(gpt_ind, actions, salience_levels,),
                    batch_idx=batch_idx,
                )
            self.gpt.global_step = self.global_step

            # Visualization steps
            if 'ZERO_LR' in os.environ and self.is_single_token2:
                test_logits, _ = self.gpt.forward(gpt_x, z_q_ind_x, a_x)
                test_target = self.gpt.s_i_to_as_i(z_q_ind_y, a_x)
                self.viz_gpt(test_logits, test_target, state=s, batch_idx=79797979)
                self.viz_simple_predict_trajectory(gpt_x, z_q_ind_x, a_x, gpt_ret, batch_idx)

            if (
                self.should_viz_predict_trajectory and
                len(self.recent_experience) >= self.gpt_seq_len and
                batch_idx % self.predicted_trajectories_size == 0 and
                self.is_single_token2  # TODO: Support this by getting gpt_x
            ):
                self.viz_simple_predict_trajectory(gpt_x, z_q_ind_x, a_x, gpt_ret, batch_idx)
            if batch_idx % self.predicted_trajectories_size == 0 \
                    and self.is_single_token2:  # TODO: Support patch based by getting `s`
                self.viz_gpt(gpt_ret['logits'], gpt_ret['target_idx'], state=s, batch_idx=batch_idx)

                # Also try this in a no_train
                # test_logits, _ = self.gpt.forward(gpt_x, z_q_ind_x, a_x)
                # test_target = self.gpt.s_i_to_as_i(z_q_ind_y, a_x)
                # self.viz_gpt(test_logits, test_target, state=s, batch_idx=batch_idx, postfix='_forward_only')

                self.viz_movie(batch_idx)
            if self.global_step == 0:
                # TODO: Make sure to call again if we update clusters as visualizing predicted trajectories
                #   uses the cluster images
                self.viz_dvq_clusters(batch_idx)

            if (
                    ('DEBUG_TRAJ_BUFF' in os.environ or (batch_idx % self.predicted_trajectories_size == 0)) and
                    len(self.predicted_trajectory_buffer) == self.predicted_trajectory_buffer.maxlen
            ):
                self.save_predicted_trajectories(batch_idx)  # View this with viz_predicted_trajectories.py

            if batch_idx % ACC_LOG_PERIOD == 0:
                log.info(f'train replay appends {self.train_buf.length}')
                log.info(f'test replay appends {self.test_buf.length}')

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
            latent_loss, dvq_loss, recon_loss = self.training_step_dvq(batch, batch_idx)
            loss = dvq_loss
            self.log_dict({
                'lightning/dvq_recon_loss': recon_loss,
                'lightning/dvq_latent_loss': latent_loss,
            })

        self.log_dict({
            'lightning/total_reward': float(self.total_rewards[-1]) if self.total_rewards else 0,
            'lightning/avg_reward': self.avg_rewards,
            'lightning/train_loss': loss,
            'lightning/episodes': self.done_episodes,
            'lightning/episode_steps': self.total_episode_steps[-1] if self.total_episode_steps else 0,
        })

        wandb_log({'loss/train': loss})

        wandb_check_flush(batch_idx)

        # return loss
        # return OrderedDict({
        #     "loss": loss,
        #     "avg_reward": self.avg_rewards,
        # })

    def validation_step(self, batch, batch_idx):
        if self.should_train_gpt:
            if self.is_single_token2:
                gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y, s, agent_state, indices = self.gpt.get_batch_vars(batch)
                gpt_ret = self.gpt.validation_step(batch=(gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y), batch_idx=batch_idx)
            else:
                gpt_ind, actions, salience_levels = self.gpt.get_batch_vars(batch)
                gpt_ret = self.gpt.validation_step(batch=(gpt_ind, actions), batch_idx=batch_idx)
            loss = gpt_ret['loss']
            wandb_log({'loss/test': loss})
            log.info(f'Validation loss {loss}, batch length {len(batch)}')
            self.log_dict({'lightning/val_loss': loss})
        else:
            log.warning('Need to implement dvq validation')

    @no_train
    def viz_dvq_clusters(self, batch_idx):
        folder = f'{ROOT_DIR}/images/viz_clusters/{get_date_str()}_r_{RUN_ID}_b_{batch_idx}'
        self._viz_dvq_clusters(folder)
        log.info(f'Wrote dvq clusters to {folder}')

    @no_train
    def viz_dvq_clusters_knn_standalone(self):
        from sklearn.neighbors import NearestNeighbors
        emb = self.dvq.quantizer.embed_code(torch.tensor(list(range(self.dvq_num_embeddings))))
        n_neighbors = 50
        nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(emb)
        distances, indices = nbrs.kneighbors(emb)
        num_clusters_per_image = 100
        im = None
        clusters_in_im = 0
        save_i = 0
        for cluster in indices:
            imz = self.dvq.decode_flat(emb[cluster], output_proj=self.dvq.quantizer.output_proj)
            imz = imz.squeeze().permute(2, 0, 3, 1).clamp(0, 1)
            imz = imz.reshape(imz.shape[0], imz.shape[1] * imz.shape[2], -1)  # align side by side
            imz = imz.detach().cpu().numpy()
            if clusters_in_im == 0:
                im = imz
            elif clusters_in_im < num_clusters_per_image:
                im = np.concatenate((im, imz), axis=0)
            clusters_in_im += 1
            if clusters_in_im == num_clusters_per_image:
                clusters_in_im = 0
                ims = Image.fromarray(np.uint8(im * 255))
                folder = f'{ROOT_DIR}/images/dvq_clusters_knn/{DATE_STR}'
                os.makedirs(folder, exist_ok=True)
                im_name = f'{folder}/dvq_clusters_knn_{n_neighbors}n_{save_i}.png'
                ims.save(im_name)
                log.info(f'Saved to {im_name}')
                save_i += 1

    @no_train
    def compress_dvq_clusters(self):
        """
        emb -25 1169 /home/a/src/learnmax/images/compress_dvq_clusters_emb/2022_04_16_10_21_51_906093/compress_dvq_clusters_pref_-25_0.png
        emb -40 758 /home/a/src/learnmax/images/compress_dvq_clusters_emb/2022_04_16_10_01_21_957291/compress_dvq_clusters_pref_-40_0.png


        """
        from sklearn.cluster import AffinityPropagation
        emb = self.dvq.quantizer.embed_code(torch.tensor(list(range(self.dvq_num_embeddings))))
        imgs = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)
        W = imgs[0].shape[1]
        assert W == imgs[0].shape[2], 'Assuming square images'
        preference = -25  # -62 has 7 bad clusters (actually garbage), -50:12, -75:garbage, -25 great!
        mode = 'emb'
        log.info(f'Compressing clusters in {mode} mode at preference {preference}')
        if mode == 'pixel':
            af = AffinityPropagation(preference=preference, random_state=0).fit(imgs.reshape(imgs.shape[0], -1))
        elif mode == 'emb':
            af = AffinityPropagation(preference=preference, random_state=0).fit(emb)
        else:
            raise ValueError('no such mode')
        cluster_centers_indices = af.cluster_centers_indices_
        log.info(f'num clusters {len(cluster_centers_indices)}')
        labels = af.labels_

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)
        clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: -len(item[1]))}  # Sort by longest first
        longest = len(list(clusters.items())[0][1])
        im = None
        clusters_in_im = 0
        save_i = 0
        num_clusters_per_image = 100
        # Save dvq checkpoint name, af.labels, af.cluster_centers
        folder = f'{ROOT_DIR}/images/compress_dvq_clusters_{mode}/{DATE_STR}'
        os.makedirs(folder, exist_ok=True)
        json_name = f'{folder}/compress_dvq_clusters_pref_{preference}.json'
        json.dump({'dvq_checkpoint': self.dvq_checkpoint,
                   'labels': [int(x) for x in af.labels_],
                   'cluster_centers': [int(x) for x in af.cluster_centers_indices_]}, open(json_name, 'w'),
                  indent=2)
        log.info(f'Saved to {json_name}')
        for cluster_i in clusters:
            im_row = None
            for img_i in clusters[cluster_i]:
                img = imgs[img_i]
                img = img.squeeze().permute(1, 2, 0).clamp(0, 1)
                img = img.detach().cpu().numpy()
                if im_row is None:
                    im_row = img
                else:
                    im_row = np.concatenate((im_row, img), axis=1)

            pad = longest - im_row.shape[1] // W
            zeros = np.zeros((W, pad * W, 3))
            im_row = np.concatenate((im_row, zeros), axis=1)
            if im is None:
                im = im_row
            else:
                im = np.concatenate((im, im_row), axis=0)

            clusters_in_im += 1
            if clusters_in_im == num_clusters_per_image:
                clusters_in_im = 0
                ims = Image.fromarray(np.uint8(im * 255))
                im = None
                im_name = f'{folder}/compress_dvq_clusters_pref_{preference}_{save_i}.png'
                ims.save(im_name)
                log.info(f'Saved to {im_name}')
                save_i += 1


    @no_train
    def viz_all_dvq_clusters_standalone(self):
        self.cuda()
        wandb.init(entity='crizcraig', mode='disabled')
        self._viz_dvq_clusters(per_cluster_folder=None)

    def _viz_dvq_clusters(self, per_cluster_folder=None):
        n = self.dvq_num_embeddings
        width = int(n ** 0.5)  # number of clusters to display horizontally
        col_i = 0
        row = []
        rows = []
        if per_cluster_folder is not None:
            os.makedirs(per_cluster_folder, exist_ok=True)
        for i in range(n):
            _i = 3070 if 'FAKE_Z_Q_EMB' in os.environ else i
            emb = self.dvq.quantizer.embed_code(torch.tensor([_i]).cuda())  # TODO: Speed up with batching
            img_t = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)
            img_t = img_t.squeeze().permute(1, 2, 0).clamp(0, 1)
            img_np = img_t.detach().cpu().numpy()
            row.append(img_np)
            if per_cluster_folder is not None:
                im = Image.fromarray(np.uint8(img_np * 255))
                im.save(f'{per_cluster_folder}/{str(i).zfill(math.ceil(math.log10(n)))}.png')
            col_i += 1
            if col_i > width:
                col_i = 0
                rows.append(np.concatenate(row, axis=1))
                row = []
        last_row = np.concatenate(row, axis=1)
        last_row_remain = rows[-1].shape[1] - last_row.shape[1]
        last_row_padded = np.concatenate((last_row, np.zeros((rows[-1].shape[0], last_row_remain, 3))), axis=1)
        rows.append(last_row_padded)
        x_hat = np.array(rows).reshape((row[0].shape[0] * width, -1, 3))  # 16 (4,68,3)
        im = Image.fromarray(np.uint8(x_hat * 255))
        # im.show()
        im.save(f'{ROOT_DIR}/images/all_dvq_clusters_{DATE_STR}.png')

    @no_train
    def viz_dvq_standalone(self, num_cols=20):
        """
        Visualize num_cols images and their reconstructions
        """
        self.cuda()
        wandb.init(entity='crizcraig', mode='disabled')

        test_loader = self.train_dataloader()

        # Forward through a few to get different trajectories
        xl = next(iter(test_loader))[0].cuda()
        xl = next(iter(test_loader))[0].cuda()
        xl = next(iter(test_loader))[0].cuda()
        xl = next(iter(test_loader))[0].cuda()
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

        xcols = torch.cat([x[:num_cols], x_hat[:num_cols]], axis=2)  # side by side x and xhat
        xrows = torch.cat([xcols[i] for i in range(num_cols)], axis=2)

        # TODO: Just save image with PIL here
        plt.figure(figsize=(num_cols, 5))
        plt.imshow((xrows.data.cpu().permute(1, 2, 0)).clamp(0, 1))
        plt.axis('off')
        plt.show()

        plt.show()

    @no_train
    @best_effort
    def viz_gpt(self, logits, target_idx, state, batch_idx, postfix=None):
        postfix = postfix or ''
        batch_index = 0  # Viz first batch (set ALWAYS_SAMPLE_LATEST) to compare with simple predict trajectory
        seq_win_size = target_idx.shape[-1]
        num_imgs_to_viz = min(10, seq_win_size)
        top_n = 3
        imgs = state[batch_index][-num_imgs_to_viz:]  # Check end of sequence

        logits = logits.detach().cpu().numpy()[batch_index][-num_imgs_to_viz:]

        # dvq states are action agnostic, so get base state by dividing by num_actions
        target_idx = self.gpt.as_i_to_s_i(target_idx[batch_index][-num_imgs_to_viz:])

        emb = self.dvq.quantizer.embed_code(target_idx)
        imgs_target = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)

        # top n predicted next embedding indexes for each img - TODO: Use torch.topk here which is already sorted
        top_n_idxs = np.argpartition(logits, -top_n)[:, -top_n:]  # top n idxs unsorted
        top_n_idxs_idxs_sorted = np.argsort(np.take_along_axis(-logits, top_n_idxs, axis=1))  # -logits => sort desc
        top_n_idxs = np.take_along_axis(top_n_idxs, top_n_idxs_idxs_sorted, axis=1)  # top n idxs sorted desc
        top_n_idxs = torch.Tensor(top_n_idxs).int()

        # get dvq embeddings from indexes
        idx_hat_state = self.gpt.as_i_to_s_i(top_n_idxs.to(target_idx.device))
        emb_hat = self.dvq.quantizer.embed_code(idx_hat_state)

        # decode predictions into images
        imgs_hat = self.dvq.decode_flat(emb_hat, output_proj=self.dvq.quantizer.output_proj)
        dim_len = len(imgs_hat.size())
        imgs_hat = imgs_hat.permute(*list(range(dim_len-3)), dim_len-2, dim_len-1, dim_len-3).clamp(0, 1)
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
        folder = f'{ROOT_DIR}/images/viz_gpt/{DATE_STR}r_{RUN_ID}_{get_date_str()}'
        os.makedirs(folder)
        filename = f'{folder}/viz_gpt_b_{batch_idx}{postfix}.png'
        log.info(f'Saving gpt viz to {filename}')
        im.save(filename)
        # wandb_log({'train/gpt-viz': [wandb.Image(im)]})  # Just save locally

            # fig1 = plt.figure(figsize=(num_imgs_per_row, 1))
            # ax1 = fig1.add_subplot(111)
            # ax1.imshow(imgs_display, interpolation='none')
            # plt.show()
            # TODO: Visualize image with action, cluster index + distance, probability, uncertainty
            #   Make sure you're extracting the logits correctly for the image
            #   Look at disappear and teleport images bad clustering cases.

    @best_effort
    def viz_movie(self, batch_idx):
        os.makedirs(f'{ROOT_DIR}/images/viz_movie', exist_ok=True)
        folder = f'{ROOT_DIR}/images/viz_movie/{get_date_str()}_r_{RUN_ID}_b_{batch_idx}'
        os.makedirs(folder, exist_ok=True)

        length = min(len(self.test_buf), 1000)
        experiences = self.test_buf.get(self.test_buf.length - length, length=length, device=self.device)
        viz_experiences(experiences, folder, self.dvq.decoder, self.device)
        # TODO: ffmpeg
        log.info(f'Saved movie to {folder}')

    @no_train
    @best_effort
    def viz_simple_predict_trajectory(self, gpt_x, z_q_ind_x, a_x, gpt_ret, batch_idx):
        # TODO: Delete/disable as viz_predicted_trajectories is better than this (created for debugging specific issue)
        parent_folder = f'{ROOT_DIR}/images/viz_simple_predict_trajectory'
        os.makedirs(parent_folder, exist_ok=True)
        folder = f'{parent_folder}/{get_date_str()}_r_{RUN_ID}_b_{batch_idx}'
        os.makedirs(folder, exist_ok=True)

        num_predict = self.num_search_steps
        sync_to_latest_batch = True
        if 'VIZ_OLD_TRAJECTORY' in os.environ:
            raise NotImplementedError('New disk based viz trajectory not implemented, use train_buf or ' +
                                      'test_buf instead of recent_experience')
            # start_delay = len(self.recent_experience) - num_predict * (num_viz - 1)
        elif sync_to_latest_batch:
            start_delay = 3
        else:
            start_delay = 0
        num_viz = (len(self.recent_experience) - start_delay) // num_predict
        num_viz = min(num_viz, 10)
        if num_viz < 1:
            log.warning('Not enough buffer to predict')
            return
        for i in range(num_viz):
            self._viz_simple_predict_trajectory(
                folder,
                num_predict,
                # delay 3 to sync with the latest train batch and compare
                delay=start_delay + i * num_predict,
                gpt_x=gpt_x,
                z_q_ind_x=z_q_ind_x,
                a_x=a_x,
                gpt_ret=gpt_ret)

    def _viz_simple_predict_trajectory(self, folder, num_predict, delay, gpt_x, z_q_ind_x, a_x, gpt_ret):
        if not self.is_single_token2:
            log.warning('Viz simple trajectory not yet implemented for patches')
            return
        # Seed with recent actions, new_states from buffer
        (action_branches,
         just_died,
         look_back,
         z_q_embed_branches,
         z_q_ind_branches,
         replay_index_branches) = self.get_recent_experience(delay=delay)

        if just_died:
            return

        truncate = self.gpt_seq_len
        # TODO: Check logit/prob KL divergence from train step to the forward here and visualize it as a histogram
        for p_i in range(num_predict):
            # SINGLE TOKEN ONLY
            logits, expected_deviation = self.gpt.forward(
                z_q_embed_branches[:, -truncate:],
                z_q_ind_branches[:, -truncate:],
                # No need to shift actions as we use new_state from buffer, also see: sa2as()
                action_branches[:, -truncate:],
            )
            # Equal up to 151, 25 - turns out there's interaction between samples in the batch on forward ;/
            # torch.equal(torch.topk(self.gpt_ret['logits'][0], 151)[1], torch.topk(logits[0], 151)[1])

            B, W, action_state_logits = logits.size()
            assert B == 1 and W <= truncate
            logits_last_step = logits[0][-1]
            probs = F.softmax(logits_last_step, dim=-1)
            # TODO: Get most likely state given some action
            # target_action_state_idx = torch.multinomial(probs, num_samples=2)[1]  # 2nd most likely
            target_action_state_idx = torch.argmax(logits_last_step)  # Get most likely action-state

            # Extract action and state from action_state
            action, z_q_ind = self.gpt.split_as_i(int(target_action_state_idx))
            if 'PREDICT_LEFT_RIGHT' in os.environ:
                action = p_i % 2 + 3

            z_q_flat = self.dvq.quantizer.embed_code(torch.tensor(z_q_ind).to(self.device))  # Get embedding from index

            # Autoregress: Add predictions to end of gpt input
            z_q_embed_branches = torch.cat((z_q_embed_branches, z_q_flat.reshape(1, 1, -1)), dim=1)
            action_branches = torch.cat((action_branches, torch.tensor(action).to(self.device).reshape(1, 1)), dim=1)
            z_q_ind_branches = torch.cat((z_q_ind_branches, torch.tensor(z_q_ind).to(self.device).reshape(1, 1)), dim=1)
        imgs = self.dvq.decode_flat(z_q_embed_branches, output_proj=self.dvq.quantizer.output_proj)
        imgs = imgs.permute(0, 1, 3, 4, 2)  # CHW => HWC
        num_actual = 5
        num_out = num_actual + num_predict
        imgs = imgs[:, -num_out:]  # Show 5 actual frames followed by num_search_levels predicted
        sz = imgs.size()
        assert sz[0] == 1
        imgs = imgs.reshape(sz[1], sz[2], *sz[3:])
        imgs = np.uint8(imgs.clamp(0, 1).detach().cpu().numpy() * 255)
        imgs = np.concatenate(imgs, axis=1)  # Align side by side
        width = int(imgs.shape[1] / (num_predict + num_actual))
        actions = action_branches[0][-num_out:]
        frames = []
        for i in range(num_out):
            frame_im = imgs[:, width * i: width * (i+1)]
            txt_im = get_np_txt_caption2(frame_im, get_action_text(self.env, actions[i]), size=42)
            if i == num_actual - 1:
                txt_im[:,-10:] = np.zeros((len(txt_im), 10, 3))
            frame_im = np.concatenate((frame_im, txt_im), axis=0)
            # concat img along height dim
            frames.append(frame_im)

        im = Image.fromarray(np.concatenate(frames, axis=1))  # Align side by side
        # im.show()
        im_name = f'{folder}/viz_simple_predict_trajectory_num_actual_{num_actual}_num_predict_{num_predict}_delay_{delay}.png'
        im.save(im_name)
        print('Saved', im_name)

    def save_predicted_trajectories(self, batch_idx):
        """
        Saves predicted trajectory for each state in the buffer. Later we view this trajectory
        with viz_predicted_trajectories.py in pygame where we use the z_q_ind to get each cluster
        image in the trajectory.
        """
        os.makedirs(f'{ROOT_DIR}/images/viz_traj', exist_ok=True)
        folder = f'{ROOT_DIR}/images/viz_traj/{get_date_str()}_r_{RUN_ID}_b_{batch_idx}'
        os.makedirs(folder, exist_ok=True)
        traj_len = len(self.predicted_trajectory_buffer)
        for i, traj in enumerate(self.predicted_trajectory_buffer):
            # Write JSON with trajectories
            state = self.recent_experience[-traj_len + i].state
            if traj is None:
                log.debug('No predicted trajectory on death')
                continue
            elif traj['z_q_ind'][0] != state.dvq_z_q_ind[0]:
                log.error('Buffer and predicted trajectory out of sync, skipping trajectory')
                log.error(f"traje {traj['z_q_ind'][0]}")
                log.error(f"state {state.dvq_z_q_ind[0]}")
                continue
            traj = {k: v.cpu().numpy().tolist() for (k, v) in traj.items()}
            str_i = str(i).zfill(9)
            json.dump(traj, open(f'{folder}/traj_{str_i}.json', 'w'))
            get_image_from_state(state.state, folder, i)  # TODO: We save this for movie too, reuse data?
            # im.show()

    def training_step_dvq(self, batch, batch_idx):
        self.dvq.set_do_kmeans(True)

        # _, agent_state = self.dvq.get_batch_vars(batch)

        # TODO: No need to call training_step again, just need to get the average loss from the batch.
        # dvq_loss = agent_state["dvq_loss"].mean()
        # recon_loss = agent_state["dvq_recon_loss"].mean()
        # latent_loss = agent_state["dvq_latent_loss"].mean()

        dvq_loss, recon_loss, latent_loss, x_hat = self.dvq.training_step(batch, batch_idx)

        # We use multiple optimizers so need to manually backward each one separately
        self.dvq_optimizer.zero_grad()
        self.manual_backward(dvq_loss)
        self.dvq_optimizer.step()
        loss = dvq_loss
        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)
        self.dvq.global_step = self.global_step
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
        if self.should_train_gpt:
            for prop_name in dir(self.dvq):
                prop = getattr(self.dvq, prop_name)
                if isinstance(prop, nn.Module):
                    prop.zero_grad()
            batch_size = self.gpt_batch_size
            self.dvq.set_do_kmeans(False)
            if self.should_overfit_gpt:
                self.warm_start_size = self.replay_buffers.overfit_length
            else:
                self.warm_start_size = 2 * self.gpt_batch_size * self.gpt_seq_len  # One val and one train batch

        else:
            self.warm_start_size = self.dvq_batch_size
            batch_size = self.dvq_batch_size  # Already set in __init__ but in case we toggle training gpt, set here too

        log.info(f'Populating replay buffer with {self.warm_start_size} experiences...')
        self.populate(self.warm_start_size)
        log.info(f'...finished populating replay buffer')

        # train_batch calls the model to get the next action, so each worker has a copy of the model!
        self.dataset = ExperienceSourceDataset(self.training_batch_gen)

        if not self.pin_memory:
            log.warning('Pin memory is off, data loading will be slow. Turn on when using disk backed replay buffer')
        return DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.num_workers,
                          drop_last=True, pin_memory=self.pin_memory)  # Images on GPU already so pin_memory raises exception

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def val_dataloader(self) -> DataLoader:
        """Get validation loader"""

        validation_dataset = ExperienceSourceDataset(self.validation_batch_gen)

        if not self.pin_memory:
            log.warning('Pin memory is off, data loading will be slow. Turn on when using disk backed replay buffer')

        if self.should_train_gpt:
            batch_size = self.gpt_batch_size
        else:
            batch_size = self.dvq_batch_size  # Already set in __init__ but in case we toggle training gpt, set here too

        # Images on GPU already so pin_memory raises exception
        return DataLoader(dataset=validation_dataset, batch_size=batch_size, num_workers=self.num_workers,
                          drop_last=True, pin_memory=self.pin_memory, shuffle=False)

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

    def tree_search(self):
        """
        Search through tree of possible actions and resulting latent states with GPT-based dynamics

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

        Algo steps:
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

        ### Action selection per saliency level
        I'm hoping there's a natural tendency to prefer more salient action exploration due to the requirement
        entropy is low before stacking new (higher) salience levels. If all the sudden, however, entropy appears at a lower level,
        there perhaps should be some preference towards higher salience if the two salience levels have
        equal entropy otherwise. This shows up in humans in language learning, where at a young age we
        are more attuned to the more granular sounds while we learn the phonemes of our language. Then as we graduate
        to higher levels of abstraction in phrases, sentences, etc... we are more interested and attuned to this space
        than phonemes which is good for exploring longer term problems. A downside of this is that
        we have a harder time learning new languages.

        ----------------------
        ## Saliency level design

        There should be context tokens: (previous salient), goal (next salient) state, and saliency level embedding
        added to each input token to the transformer. Also try splitting these across heads instead of adding so
        that we can pass zeroes to them when we don't have context and reuse the learning that takes place without
        context.

        The salient states are just dvq outputs, i.e. z states at each saliency level, which allows the same
        transformer to be reused for each saliency level. The difference is that, the context token will dictate
        skipping lower level states directly to the next salient state at that level. The distance between salient
        states is variable, e.g. "first i'll climb this mountain (for 1 hour), then I'll take a picture (2 minutes),
        and eat lunch (20 minutes)."

        We need to stop adding levels when we run out of memory. cf visual cortex levels => prefrontal cortex

        I think we'll have to pass zeroes in for the next salient when we are starting out (or just don't add it).
        We only know the goal state when we've searched for the most interesting path and we know the next salient
        state on the way at the current level.

        We need to know when a goal state is no longer likely due to unpredicted changes in lower level action-states.
        Two ways we can do this are 1) predict time-to-salient as a GPT output and/or 2) use lower level context tokens
        as well. The problem with 2) is that we need to maintain a stable prediction with all possible low level
        states as context which seems like it would greatly reduce sample efficiency.

        To solve this, append higher and lower level context tokens as special input tokens to the
        front of the window, rather than summing them with the current tokens like position, action, etc... This as
        it allows all heads to see it without prior connections to other token positions needing to change when the
        context changes / is learned. Initially zeros should be passed for context when salient states have not yet
        been learned. Then as salient levels are formed, the context token can just be the logits (output) of the
        transformer above / below the current level. This allows all tokens in the window adjacent to a level to
        inform its predictions. We still need a goal salient state, which we are pursuing for entropy, and this goal
        could be summed with the logits or provided as a separate equally sized token. However, once the goal salient
        state is no longer the most likely next state, we need to rerun tree search at that level. We may still also need
        to output a max number of low level states we expect to see before a given next state, after which the state
        should no longer be considered a valid goal state. Hopefully the low level context confers this though.

        To simplify the above, let's remove the low level context in the form of logits delivered to the higher level.
        All we should need is the expected number of low level steps required to reach the next salient state in the
        plan output by tree search. Then we should replan at that salient level with a probability based on how
        far past the expected number of states we've reached. E.g. if we expect to reach the next salient state
        in 10 steps, but have not seen *any* salient step by 20 steps, then the probability of replanning is
        1 = max(0, (actual_steps - expected_steps) / expected_steps )
        If at any time we see a state (i.e action-state) that is a known salient state at that level, then we
        also replan at that level as we've encountered a different sequence. The downside to this is that we don't
        have an up to date idea of whether the planned next salient is still likely given low level context,
        so actually we could simplify by removing the expected count and just use the logit context from below.



        Compression (does not apply to possibility-based saliency):
            If the next salient state is reached and it's no longer surprising, i.e. it's by far the most probable state
            and it has small reconstruction error, then we can train the level above to just skip this state as the
            current level can readily predict it. But we don't want this new transition to cause surprise that
            leads to a salient state above it. Checking to see if the predicted state (the old transition) was
            encountered at the lower level seems too complicated, so we can just avoid compression for now.
            **We do however need to keep track of when the goal state is reached and advance the saliency levels
              above when this happens**


        ## Tree search (batches are branches)

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

        We are searching greedily right now (top k every step) and just pursue the top k highest entropy actions.
        Saliency levels will mitigate the shortsightedness of this approach.

        HOWEVER, we could also add some monte-carlo rollouts in order to find states with delayed learning.

        For first level of the tree, only worry about one match (i.e. most recent action in env)
        For subsequent levels, the batch will represent different possible futures
        For ALL levels, only the last action in the window matters

        TODO: Find salient events in this search: Done???
            Look for when sets of high prob (2x random?) reachable states (not action_states) change significantly.
            So instead of just jumping up and down and coming back to the same set of reachable states over 4-5 (num_levels)
            actions, moving right allows reaching the ladder state or the falling states or the ground states.
            So the reachable states change a lot. We could look at KL divergence from the previous 4-5 to the current
            4-5 states, so long as the post-softmax entropy is adequetely low (i.e. we are confident in the states
            we can reach).

        TODO:
            Since high level novelty takes longer to reduce and high level states change less frequently,
            we can run tree search at intermediate/lower levels as they change. Then so long as we don't
            encounter low level novelty, we can continue pursuing high level novelty. But when we do encounter
            low level novelty, we should immediately sample that and learn from it. See get_recent_experience()
            Anytime a saliency level is trained, we should rerun tree search on that level to see if there's
            potential learning there that would lead us to change our plan. Such changes to intermediate level
            goals doesn't necessarily mean our high level goals need to change, but we'll need some way of knowing
            whether we are off-track wrt our high level goals by realizing goal states are no longer high probability
            outcomes of the current context - at which point we will need to replan up through all saliency levels.

        TODO:
            Really what we want is to follow high probability / high saliency
            paths until some level of uncertainty is encountered.
            Right now we just greedily take uncertain actions. This is fine if entropy is high, but if entropy is low
            for the initial action, most learning will occur later on.
            We want to sample such that possibilities are expanded. To do this we'll follow
            the highest known high (high probability) salient states trajectories which should be low entropy.
            Then when we reach the frontier of knowledge, we see high entropy and unknown possibilities.
            We must therefore clarify the knowledge frontier (i.e. resolve uncertainty) to expand possibilities
            at the knowledge frontier. Possibilities / opportunities at higher saliency levels, represent longer term
            and wider reaching states that will encompass learning at all the levels below (e.g. humans on mars will
            mean learning from the vastly different sensory experience on mars up through different day/night, month,
            year, etc... patterns.)

        TODO:
            The entropy score of a plan should be weighted by the joint action-state probabilities that
            make up the path multiplied by the entropy at the end of the path. We could also continue to sum
            the entropy along the path and multiply by that. When we are using salient states though, entropy along
            the path will be low until we get to the frontier of knowledge, so most entropy should come from
            the end anyway.

        """
        beam_width = self.beam_width
        num_steps = self.num_search_steps
        # TODO:
        #   - If search finds termination, trim from tree. But what if all branches terminate - should not be possible
        #       really need to confirm though.

        # Things to optimize if this is too slow (on GPU it's 100ms now with a 3090 so it's okay to play real time
        #   at least)
        #  1. optimize gpt forward
        #  2. do 3 to 5 step look ahead and rely on saliency for long term planning
        #  3. increase beam width for better quality look ahead (mostly parallel so is fast)
        (action_branches,
         just_died,
         look_back,
         z_q_embed_branches,
         z_q_ind_branches, replay_index_branches) = self.get_recent_experience()

        if just_died:
            return None, None

        # Unsqueeze so we have batch, window, embed dims, where batch=1
        entropy_branches = torch.tensor([0]).unsqueeze(0).to(self.device)  # Total entropy of branch

        assert z_q_embed_branches.size()[0] == 1, 'We should only start with one trunk'
        action_entropy_path = None  # Declare for output
        num_steps -= 1  # first step already populated
        for step in range(num_steps):
            should_log_wandb = (step == num_steps - 1) and self.global_step % WANDB_MAX_LOG_PERIOD == 0

            # Forward predict one step in the search tree
            # as_ind = self.gpt.s_i_to_as_i(z_q_ind_branches, action_branches)  # action, state => action_state
            logits, expected_deviation = self.gpt.forward(
                z_q_embed_branches[:, -self.gpt_seq_len:],
                z_q_ind_branches[:, -self.gpt_seq_len:],
                # No need to shift actions as we use new_state from buffer, also see: sa2as()
                action_branches[:, -self.gpt_seq_len:]
            )
            B, W, _ = z_q_embed_branches[:, -self.gpt_seq_len:].size()  # batch size, window size

            # Get entropy across actions, see GPT.s_i_to_as_i() for ordering
            logits = logits.reshape(B, W, -1, self.num_actions)  # B, W, S * A => # B, W, S, A where S = dvq state index
            # logits = logits.reshape(B, W, self.num_actions, -1)  # Trying this idk why
            if should_log_wandb:
                self.log_p_a_given_rs(logits)
            logits = logits.transpose(-1, -2)  # B, W, S, A => B, W, A, S

            # We only care about next step uncertainty/entropy, so omit all but last step from window
            logits = logits[:, -1:, ...]
            assert logits.size()[1] == 1   # assert W == 1 now
            probs = F.softmax(logits, dim=-1)  # Probability of dvq clusters given actions

            if should_log_wandb:
                # Log p(s|a)
                max_p_s_given_a = probs.max(dim=-1)
                wandb_log({f'max p(s|a)/max_a': max_p_s_given_a.values.max()})
                wandb_log({f'max p(s|a)/mean_a': max_p_s_given_a.values.mean()})
                wandb_log({f'max p(s|a)/min_a': max_p_s_given_a.values.min()})

            # TODO: Use bayes to verify logit normalization across states = actions

            # Action entropy across states
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)

            if should_log_wandb:
                wandb_log({f'entropy/action min': entropy.min()})
                wandb_log({f'entropy/action mean': entropy.mean()})
                wandb_log({f'entropy/action max': entropy.max()})

            # Add entropy gathered thus far in branches to path entropy of most recent action
            entropy_total = entropy_branches[:, -1]  # Summed along the way so last index has total

            # Broadcast to all actions - B => B, A
            entropy_total = entropy_total.unsqueeze(1)
            B, A = entropy_total.size()[0], entropy.size()[-1]
            assert (B == 1 and step == 0) or B == beam_width, 'Unexpected num branches'
            assert A == self.num_actions, 'Unexpected action dim size'
            entropy_total = entropy_total * torch.ones((B, A), device=self.device)

            # Add entropy of latest action to totals with window dim of length 1
            entropy_total = entropy_total.unsqueeze(1) + entropy

            # Get globally interesting next actions across all branch tips,
            # some branches may have more than 1 interesting action, others 0
            actions, action_entropy_path, actions_flat = topk_interesting(entropy_total, k=beam_width,
                                                                          rand_half='RAND_HALF' in os.environ)
            if should_log_wandb:
                wandb_log({f'entropy/traj min': action_entropy_path.min()})
                wandb_log({f'entropy/traj mean': action_entropy_path.mean()})
                wandb_log({f'entropy/traj max': action_entropy_path.max()})
            action_states = get_action_states(logits, actions_flat)

            # Branch out by copying beginning of path for each new action state and appending new action state to end
            new_z_q_ind_branches = []
            new_z_q_embed_branches = []
            new_action_branches = []
            new_entropy_branches = []
            for i, (bi, ai) in enumerate(actions):  # batches are branches
                # TODO: Vectorize this loop for high beam sizes (>10?)

                # Get heads
                z_q_ind_head = z_q_ind_branches[bi]
                z_q_embed_head = z_q_embed_branches[bi]
                action_head = action_branches[bi]
                entropy_head = entropy_branches[bi]

                # Get tails
                z_q_ind_tail = action_states[i]
                z_q_embed_tail = self.dvq.quantizer.embed_code(z_q_ind_tail)
                action_tail = ai
                entropy_tail = action_entropy_path[i]

                # Add head+tail to new branches
                new_z_q_ind_branches.append(torch.cat((z_q_ind_head, z_q_ind_tail.unsqueeze(0))))
                new_z_q_embed_branches.append(torch.cat((z_q_embed_head, z_q_embed_tail.unsqueeze(0))))
                new_action_branches.append(torch.cat((action_head, action_tail.unsqueeze(0))))
                new_entropy_branches.append(torch.cat((entropy_head, entropy_tail.unsqueeze(0))))
            # Convert to tensor with window dim <= GPT block size
            z_q_ind_branches = torch.stack(new_z_q_ind_branches)
            z_q_embed_branches = torch.stack(new_z_q_embed_branches)
            action_branches = torch.stack(new_action_branches)
            entropy_branches = torch.stack(new_entropy_branches)

        # Return branches for highest entropy path
        most_interesting_i = torch.argmax(action_entropy_path)
        predicted_traj = {
            # Return current and future trajectory without past
            'action': action_branches[most_interesting_i][-num_steps-1:],  # -1 to include most recent step
            'z_q_ind': z_q_ind_branches[most_interesting_i][-num_steps-1:],  # -1 to include most recent step
            'entropy': entropy_branches[most_interesting_i],
        }

        # look_back actions are context / have already been taken so return first action after look_back
        next_action = action_branches[most_interesting_i][look_back+1]
        return [int(next_action)], predicted_traj

    def get_recent_experience(self, delay=0, max_size=None, train_buff_only=False):
        """
        Return recent embeddings, embedding indexes, and actions. May not return a full
        window if agent was recently born. (language: using "born" is better than "spawned" for promoting thinking
        more about algorithmic alignment with humans)

        delay (int): Number of frames to shift window left, leaving out most recent
        """
        # TODO: Allow sampling replay buffer based on reconstruction error for dvq or prediction error for
        #   GPT to emphasize learning surprising/novel events. We should mark such experiences for immediate
        #   inclusion into the next batch as well.
        #   Alternatively we could just keep these experiences for longer so they are sampled more throughout training.
        #   We learn more from such experiences by default due to high error, however, sample efficiency may be improved
        #   even more through sampling.

        z_q_embed_branches = []
        z_q_ind_branches = []
        action_branches = []
        replay_index_branches = []

        max_size = max_size or self.gpt_seq_len
        recent_exp = self.replay_buffers.short_term_mem
        look_back = min(len(recent_exp), max_size)
        just_died = False
        all_branches = [z_q_embed_branches, z_q_ind_branches, action_branches, replay_index_branches]
        # Populate gpt window with recent experience
        for i in reversed(range(look_back)):
            b_i = -i - 1 - delay
            if -b_i > len(recent_exp):
                # delay caused us to look back before beginning of buffer
                look_back = i
                continue
            exp = recent_exp[b_i]
            if (
                    # Don't sample across resets - important for valuing life, algorithmic alignment, AGI safety
                    # Note that in Montezuma's revenge and other Atari games, resets occur after all lives are lost :(
                    exp.done is True
                    or
                    # Don't include non-train data
                    (train_buff_only and not exp.is_train())
            ):
                look_back = i
                # Void beginning of branches so we don't mix lives/splits
                [b.clear() for b in all_branches]
                continue
            else:
                # Use new state in single-token as outputs are action-states, i.e. the agent took an action and
                # ended up in state. We do this in order to predict the outcome of taking certain actions.
                # With state-actions you could only evaluate probabilities of actions when landing in certain states.
                # In patch-based prediction, we use the default state-action order and output tokens can be patches,
                # an action, or the frame delimiter.
                state = exp.new_state if self.is_single_token2 else exp.state
                state.to_(self.device)
                z_q_embed_branches.append(state.dvq_z_q_flat)
                z_q_ind_branches.append(state.dvq_z_q_ind)
                action_branches.append(exp.action)
                replay_index_branches.append(exp.replay_index)
        if look_back == 0:
            just_died = True
            z_q_embed_branches = None
            z_q_ind_branches = None
            action_branches = None
            replay_index_branches = None
        else:
            # Unsqueeze so we have batch, window, embed dims, where batch=1, e.g. 1, 80, 4410
            z_q_embed_branches = torch.cat(z_q_embed_branches).unsqueeze(0)
            z_q_ind_branches = torch.cat(z_q_ind_branches).unsqueeze(0)
            action_branches = torch.tensor(action_branches).unsqueeze(0).to(self.device)
            replay_index_branches = torch.tensor(replay_index_branches).unsqueeze(0).to(self.device)
        return action_branches, just_died, look_back, z_q_embed_branches, z_q_ind_branches, replay_index_branches

    def log_p_a_given_rs(self, logits):
        """
        Takes B,W,S,A logits and logs probability of action across states

        Note this is action given the resulting state, not the typical action given current state.

        I'm doing this to try and understand why conditional entropy does not drop while action-state entropy does.
        """
        probs = F.softmax(logits, dim=-1)  # Probability of action given resulting state
        max_p_a_given_rs = probs.max(dim=-1)  # Max prob action for each state
        wandb_log({f'max p(a|rs)/max_rs': max_p_a_given_rs.values.max()})
        wandb_log({f'max p(a|rs)/mean_rs': max_p_a_given_rs.values.mean()})
        wandb_log({f'max p(a|rs)/min_rs': max_p_a_given_rs.values.min()})

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
    #       # implement your own custom logic to clip gradients for generator (optimizer_idx=0)

LOAD_LAYER_TYPES = (nn.ConvTranspose2d, ResBlock, nn.ReLU, nn.Conv2d)


def set_net_weights(source, target):
    # TODO: Delete as this is handled by load_state_dict
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


def cli_main(get_model_args_fn=get_model_args_from_cli, get_train_args_fn=get_train_args_from_cli):
    torch.backends.cudnn.benchmark = True  # autotune kernels
    torch.multiprocessing.set_start_method('spawn')
    open('/tmp/learnmax-runs.txt', 'a').write(f'{DATE_STR} {" ".join(sys.argv)}\n')
    args = get_model_args_fn()
    viz_dvq = args.viz_dvq or args.viz_all_dvq_clusters or args.viz_dvq_clusters_knn or args.compress_dvq_clusters

    if viz_dvq:
        args.should_train_gpt = False
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
    del learn_max_args['viz_dvq']  # Not for model
    del learn_max_args['viz_all_dvq_clusters']  # Not for model
    del learn_max_args['viz_dvq_clusters_knn']  # Not for model
    del learn_max_args['compress_dvq_clusters']  # Not for model
    del learn_max_args['checkpoint']  # Not a model property
    if 'viz_predict_trajectory' in learn_max_args:
        learn_max_args['should_viz_predict_trajectory'] = learn_max_args['viz_predict_trajectory']
        del learn_max_args['viz_predict_trajectory']  # Changed the name for class

    if args.checkpoint:
        assert not args.dvq_checkpoint, ('dvq_checkpoint and checkpoint are mutually exclusive, ' +
                                         'set dvq_checkpoint if you want to train gpt from scratch')
        # model = LearnMax.load_from_checkpoint(args.checkpoint)
        model = LearnMax.load_from_checkpoint(args.checkpoint, **learn_max_args)
        # load_pretrained_dvq_and_gpt(args, model)
    else:
        model = LearnMax(**learn_max_args)

    if args.dvq_checkpoint:
        assert not args.checkpoint, ('dvq_checkpoint and checkpoint are mutually exclusive, ' +
                                     'set checkpoint if you want to load gpt and dvq')
        load_pretrained_dvq(args, model)

    if args.viz_dvq:
        return model.viz_dvq_standalone()
    elif args.viz_all_dvq_clusters:
        return model.viz_all_dvq_clusters_standalone()
    elif args.viz_dvq_clusters_knn:
        return model.viz_dvq_clusters_knn_standalone()
    elif args.compress_dvq_clusters:
        return model.compress_dvq_clusters()
    else:
        delattr(args, 'viz_dvq')
        delattr(args, 'viz_all_dvq_clusters')
        delattr(args, 'viz_dvq_clusters_knn')
        delattr(args, 'compress_dvq_clusters')

    args = get_train_args_fn()

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

    if model.should_train_gpt:
        train_gpt(args, fast_dev_run, model)
    else:
        train_dvq(args, model, wandb_logger, fast_dev_run)


def train_dvq(args, model, wandb_logger, fast_dev_run):
    # -------------------- Standard dvq training
    # annealing schedules for lots of constants
    checkpoint_callback = ModelCheckpoint(monitor='lightning/dvq_recon_loss', mode='min', every_n_train_steps=1000,
                                          save_top_k=1,
                                          verbose=True)
    callbacks = [checkpoint_callback,
                 DecayLR()]
    if False and args.dvq_vq_flavor == 'gumbel':  # Not used yet
        callbacks.extend([DecayTemperature(), RampBeta()])
    # Things to try
    # More steps cos anneal goes to 1.2M! Not helping
    # Fewer clusters, we're only using ~50 out of 4096: Resulted in blurry images
    # Do k-means throughout training, not just once: works great! went from 10/20 to 19.5/20 correct images
    # Output image, we can count images that are obviously wrong
    # 10 points per cluster seemed to work better than 20, try 15, etc...: 15 works pretty well vs 20 and 10

    # Max steps can be like 5k I think.
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=int(1.2e6),
                                            logger=wandb_logger, gpus=args.num_gpus, fast_dev_run=fast_dev_run)
    trainer.fit(model)


def train_gpt(args, fast_dev_run, model):
    # Train GPT
    log.info("preparing the learning rate schedule")
    # number of tokens backpropped in one iteration, need to reduce for zuma vs char as tokens are 10x larger
    iter_tokens = model.gpt_batch_size * model.gpt_seq_len
    epoch_tokens = math.ceil(model.batches_per_epoch * iter_tokens)
    log.warning('using hardcoded warmup tokens')
    lr_decay = GptWarmupCosineLearningRateDecay(learning_rate=model.gpt_learning_rate,
                                                warmup_tokens=512 * 20,  # epoch_tokens // 2,
                                                final_tokens=args.num_epochs * epoch_tokens)
    t0 = time.time()
    log.info(f'training for {args.num_epochs} epochs...')
    checkpoint_callback = ModelCheckpoint(monitor='lightning/val_loss', mode='min', every_n_train_steps=1,
                                          save_top_k=3,
                                          verbose=True)
    if os.getenv('CUDA_VISIBLE_DEVICES') == '':
        args.num_gpus = 0
    deterministic = True
    if deterministic:
        log.warning('Deterministic is on, turn off to speed up')
    trainer = pl.Trainer(gpus=args.num_gpus,
                         max_epochs=args.num_epochs,
                         # gradient_clip_val=1.0, # lightning does not support with manual optimization which we need due to two optimizers
                         callbacks=[lr_decay, checkpoint_callback],
                         precision=args.precision,
                         default_root_dir=args.default_root_dir,
                         # logger=wandb_logger,  # Don't do this as we have special throttling for wandb
                         deterministic=deterministic,  # Turn off deterministic to speed up
                         # overfit_batches=1,
                         fast_dev_run=fast_dev_run,
                         num_sanity_val_steps=0,
                         val_check_interval=10,
                         # Ensure we have data before validating
                         )
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


def load_pretrained_dvq_and_gpt(args, model):
    map_location = None
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        map_location = torch.device('cpu')
    _model = torch.load(args.checkpoint, map_location=map_location)
    _load_pretrained_dvq(_model, model)
    _load_pretrained_gpt(_model, model)


def load_pretrained_dvq(args, model):
    map_location = None
    if not os.getenv('CUDA_VISIBLE_DEVICES'):
        map_location = torch.device('cpu')
    _model = torch.load(args.dvq_checkpoint, map_location=map_location)
    _load_pretrained_dvq(_model, model)


def _load_pretrained_dvq(_model, model):
    state_dict = {k: v for k, v in _model['state_dict'].items() if k.startswith('dvq')}
    state_dict = {k[4:]: v for k, v in state_dict.items()}  # Remove `dvq.` as we're loading a submodule
    model.dvq.load_state_dict(state_dict)


def _load_pretrained_gpt(_model, model):
    state_dict = {k: v for k, v in _model['state_dict'].items() if k.startswith('gpt')}
    state_dict = {k[4:]: v for k, v in state_dict.items()}  # Remove `gpt.` as we're loading a submodule
    model.gpt.load_state_dict(state_dict)


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


def get_image_from_state(state, folder, i, should_save=True):
    state = state.squeeze(0).permute(1, 2, 0).cpu()
    im = Image.fromarray(np.uint8(state * 255))
    if should_save:
        filename = f'{folder}/{str(i).zfill(9)}.png'
        im.save(filename)
    return im


if __name__ == '__main__':
    cli_main()



