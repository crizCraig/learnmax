import argparse
import json
import math
import os
import time
from typing import Tuple, List, OrderedDict, Dict, Optional

import cv2
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import wandb
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

from learn_max.agent import LearnMaxAgent
from learn_max.utils import topk_interesting, _init_weights
from learn_max.constants import SAVE_DIR, SEED
from learn_max.dvq.vqvae import VQVAE, DecayLR, DecayTemperature, RampBeta
from learn_max.mingpt.lr_decay import WarmupCosineLearningRateDecay
from learn_max.mingpt.model import GPT


class LearnMax(pl.LightningModule):
    def __init__(
            self,
            embedding_dim: int = 512,  # length of embedding vectors output by dvq to transformers

            # TODO: Add more levels of transformers for salient events

            # dvq args - dvq = deep vector quantization
            dvq_n_hid: int = 64,  # number of channels controlling the size of the model
            dvq_num_embeddings: int = 512,  # vocabulary size; number of possible discrete states
            dvq_loss_flavor: str = 'l2',  # `l2` or `logit_laplace`
            dvq_input_channels: int = 3,  # 3 for RGB
            dvq_enc_dec_flavor: str = 'deepmind',  # Deepmind VQVAE or OpenAI Dall-E dVAE
            dvq_vq_flavor: str = 'vqvae',  # `vqvae` or `gumbel`

            # mingpt model definition args
            # size of the vocabulary (number of possible tokens) -
            #  64 for Shakespeare
            #  8,192 in DALL·E images
            #  16,384 for DALL·E words
            gpt_vocab_size: int = 64,

            # length of the model's context window in time
            gpt_block_size: int = 128,
            gpt_n_layer: int = 8,  # depth of the model; number of Transformer blocks in sequence
            gpt_n_head: int = 8,  # number of heads in each multi-head attention inside each Transformer block

            # mingpt model optimization args
            gpt_learning_rate: float = 3e-4,  # the base learning rate of the model
            gpt_weight_decay: float = 0.1,  # amount of regularizing L2 weight decay on MatMul ops
            gpt_betas: Tuple[float, float] = (0.9, 0.95),  # momentum terms (betas) for the Adam optimizer
            gpt_embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
            gpt_resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
            gpt_attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix

            # RL type stuff
            avg_reward_len: int = 100,  # how many episodes to take into account when calculating the avg reward
            epoch_len: int = 1000,  # how many batches before pseudo epoch
            gamma: float = 1,  # discount factor - only used for evaluation metrics right now
            n_steps: int = 1,  # number of steps to return from each environment at once
            replay_size: int = 100_000,  # number of steps in the replay buffer
            env_id: str = 'MontezumaRevenge-v0',  # gym environment tag
            warm_start_size: int = 10_000,  # how many samples do we use to fill our buffer at the start of training
            batches_per_epoch: int = 10_000,  # number of batches per pseudo (RL) epoch

            # Standard stuff
            num_workers: int = 0,  # data loader workers
            data_dir: str = SAVE_DIR,  # place to save tfboard logs and checkpoints
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

        # Hyperparameters
        self.dvq_embedding_dim = embedding_dim  # size of the embedding vector representing a cluster of embeddings
        self.dvq_n_hid = dvq_n_hid
        self.dvq_num_embeddings = dvq_num_embeddings
        self.dvq_loss_flavor = dvq_loss_flavor
        self.dvq_input_channels = dvq_input_channels
        self.dvq_enc_dec_flavor = dvq_enc_dec_flavor
        self.dvq_vq_flavor = dvq_vq_flavor

        self.gpt_embedding_dim = embedding_dim  # the "width" of the model (embedding_dim), number of channels in each Transformer
        self.gpt_block_size = gpt_block_size
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
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.save_hyperparameters()

        def make_env(_env_id):
            _env = gym.make(_env_id)
            _env = MaxAndSkipEnv(_env)
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

        self.state = self.env.reset()

        self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=dvq_num_embeddings, embedding_dim=self.dvq_embedding_dim,
                         loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                         enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor)
        self.gpt = GPT(vocab_size=gpt_vocab_size, block_size=gpt_block_size, n_layer=gpt_n_layer,
                       embedding_dim=self.gpt_embedding_dim, n_head=gpt_n_head, learning_rate=gpt_learning_rate,
                       weight_decay=gpt_weight_decay, betas=gpt_betas, embd_pdrop=gpt_embd_pdrop,
                       resid_pdrop=gpt_resid_pdrop, attn_pdrop=gpt_attn_pdrop)

        self.agent = LearnMaxAgent(model=self, num_actions=self.env.action_space.n)

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
                action = self.agent(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
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

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            # TODO: Subtract 0.5 from image so that we - map [0,1] range to [-0.5, 0.5]

            # TODO: Allow learning within the episode

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)

            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> OrderedDict:
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

        dvq_loss = self.dvq.training_step(batch, batch_idx)
        self.gpt.step_('train', )

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        self.log_dict({
            "total_reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
            "train_loss": loss,
            "episodes": self.done_episodes,
            "episode_steps": self.total_episode_steps[-1]
        })

        return OrderedDict[{
            "loss": loss,
            "avg_reward": self.avg_rewards,
        }]

    def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluate the agent for 10 episodes"""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        dvq_optimizer = self.dvq.configure_optimizers()
        gpt_optimizer = self.gpt.configure_optimizers()
        return [dvq_optimizer, gpt_optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, pin_memory=True)

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
            default=100_000,
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=10_000,
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
          get the action that corresponds to each output state by forwarding it through the dvq decoder. We need to
          learn an action embedding for this.
          in batch(es) and getting the closest action embedding to the decoded one
        * add deviations to _all_ encountered sorted deviations
            * try naive unoptimal way first though since we don't have that many branches)
            * If we need to optimize, use TORCH.SEARCHSORTED
                * insert new action_deviations with torch.cat - but also keep track of the associated sequence index in another
                  (unsorted) dimension of the the action_deviations pool
            * sequences just get appended to and are not sorted, so sequence index can be static.
        * get topk_interesting(action_deviations, k) output indexes - (nb. interesting is defined as within 50-75pct
          uncertainty) - however we may need to do 50-100pct to avoid state starvation
        * limit the initial actions in the trajectory to ones with a predicted state that close to the one received from the sim
        * feed the corresponding z,a embeddings at those indexes back into the transformer at the end of the
          current sequence (nb. we are simulating the env in multiple future branches)
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


def cli_main():
    parser = argparse.ArgumentParser()

    # model args
    parser = LearnMax.add_reinforcement_learning_args(parser)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True  # autotune kernels

    # common = {'batch_size': args.gpt_batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}

    model = LearnMax(**args.__dict__)


    # trainer args  # TODO: Check that our defaults above are preserved for overlapping things like pin-memory
    parser.add_argument('-x', '--num_epochs', type=int, default=2, help="number of epochs to train for")
    parser.add_argument('-b', '--gpt_batch_size', type=int, default=64, help="batch size to train gpt with")
    parser.add_argument('-l', '--gpt_block_size', type=int, default=128, help="block size for the model (length of window of context)")
    parser.add_argument('-g', '--num_gpus', type=int, default=1, help="number of gpus to train on")
    parser.add_argument('-n', '--num_workers', type=int, default=0, help="number of workers for dataloading")
    parser.add_argument('-p', '--pin_memory', type=int, default=0, help="pin memory on dataloaders?")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    wandb.init(entity='crizcraig', save_code=True)  # TODO: Try comet for code saving, it looks like it saves and does diffs on everything, not just this file
    wandb.watch(model)

    wandb_logger = WandbLogger()

    log.info(json.dumps(vars(args), indent=0))

    """
    Algo

    Fill up replay buffer for a while, taking random actions to start. .populate()

    Train the dvq on the replay buffer randomly shuffled.

    Use dvq tokens to train transformer (gpt-architecture) 
    """
    # -------------------- Standard dvq training

    # annealing schedules for lots of constants
    callbacks = [ModelCheckpoint(monitor='val_recon_loss', mode='min'), DecayLR()]
    if False and args.dvq_vq_flavor == 'gumbel':  # Not used yet
       callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=100_000_000, logger=wandb_logger)

    trainer.fit(model)


    # -------------------- Standard mingpt training
    log.info("preparing the learning rate schedule")
    iter_tokens = args.gpt_batch_size * args.gpt_block_size  # number of tokens backpropped in one iteration
    epoch_tokens = math.ceil(args.batches_per_epoch * iter_tokens)
    lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4,
                                             warmup_tokens=512 * 20,  # epoch_tokens // 2,

                                             final_tokens=args.num_epochs * epoch_tokens)
    t0 = time.time()
    log.info("training...")
    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0, callbacks=[lr_decay],
                         precision=args.precision, default_root_dir=args.default_root_dir)
    trainer.fit(model.gpt, train_dataloader, val_dataloader)  # Need to populated these
    t1 = time.time()
    log.info("%d epochs took %fs, or %fs/epoch" % (args.num_epochs, t1 - t0, (t1 - t0) / args.num_epochs))

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="max_interesting", mode="max", period=1, verbose=True)

    seed_everything(SEED)  # env is seeded later
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    # end standard mingpt training


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

if __name__ == '__main__':
    cli_main()
