import argparse
import json
import math
import os
import time
from typing import Tuple, List, OrderedDict, Dict, Optional

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import wandb
from gym import Env
from loguru import logger as log
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.datamodules.experience_source import Experience
from pl_bolts.models.rl.common.gym_wrappers import make_environment
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from learn_max.agent import LearnMaxAgent
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
        self.gpt_n_embd = embedding_dim  # the "width" of the model (embedding_dim), number of channels in each Transformer
        self.batches_per_epoch = batches_per_epoch
        self.gamma = gamma
        self.n_steps = n_steps
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size

        self.save_hyperparameters()

        self.num_workers = num_workers
        self.data_dir = data_dir

        if 'DISABLE_ENV_WRAPPERS' in os.environ:
            self.env = gym.make(env_id)
            self.env.seed(SEED)
            self.test_env = gym.make(env_id)
        else:
            # Thinking these slow things down a lot, but at what benefit?!
            self.env = self.make_environment(env_id, 0)
            self.test_env = self.make_environment(env_id)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.avg_reward_len = avg_reward_len
        self.epoch_len = epoch_len
        self.gamma = gamma

        # Tracking metrics
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states = []
        self.batch_actions = []

        self.state = self.env.reset()
        self.agent = LearnMaxAgent()

        self.dvq = VQVAE(n_hid=dvq_n_hid, num_embeddings=dvq_num_embeddings, embedding_dim=self.dvq_embedding_dim,
                         loss_flavor=dvq_loss_flavor, input_channels=dvq_input_channels,
                         enc_dec_flavor=dvq_enc_dec_flavor, vq_flavor=dvq_vq_flavor)
        self.gpt = GPT(vocab_size=gpt_vocab_size, block_size=gpt_block_size, n_layer=gpt_n_layer,
                       n_embd=self.gpt_n_embd, n_head=gpt_n_head, learning_rate=gpt_learning_rate,
                       weight_decay=gpt_weight_decay, betas=gpt_betas, embd_pdrop=gpt_embd_pdrop,
                       resid_pdrop=gpt_resid_pdrop, attn_pdrop=gpt_attn_pdrop)

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
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

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

    wandb.watch(model)

    wandb_logger = WandbLogger()

    log.info(json.dumps(vars(args), indent=0))

    """
    Algo

    Fill up replay buffer for a while, taking random actions to start.

    Train the dvq on the replay buffer randomly shuffled.

    Use dvq tokens to train transformer (gpt-architecture) 
    """
    # -------------------- Standard dvq training

    # annealing schedules for lots of constants
    callbacks = [ModelCheckpoint(monitor='val_recon_loss', mode='min'), DecayLR()]
    if args.vq_flavor == 'gumbel':  # Not used yet
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

if __name__ == '__main__':
    cli_main()
