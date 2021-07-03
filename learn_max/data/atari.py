from typing import Tuple, List

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl

from learn_max.constants import SAVE_DIR
from learn_max.dvq.data.experience_source import ExperienceSourceDataset
from learn_max.dvq.model.loss import LogitLaplace, Normal


class AtariData(pl.LightningDataModule):
    """ Provide data from Atari Learning Environment """

    def __init__(self,
                 num_workers,  # data loader workers
                 data_dir: str = SAVE_DIR,  # place to save tfboard logs and checkpoints
                 loss_flavor: str = 'l2',  # type of dvq loss  # TODO: Remove this
                 env_id: str = 'MontezumaRevenge-v0',  # gym environment tag
                 batch_size: int = 64,  # mini-batch size
                 avg_reward_len: int = 100,  # how many episodes to take into account when calculating the avg reward
                 epoch_len: int = 1000,  # how many batches before pseudo epoch
                 gamma: float = 1,  # discount factor - only used for evaluation metrics right now
                 ):
        super().__init__()
        self.inmap = {'l2': Normal.InMap(), 'logit_laplace': LogitLaplace.InMap()}[loss_flavor]


