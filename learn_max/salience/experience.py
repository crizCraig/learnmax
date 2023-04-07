from dataclasses import dataclass
from typing import Optional, Iterator

import numpy as np
import torch

from learn_max.agent import AgentState


@dataclass
class Experience:
    replay_index: Optional[int] = None  # Set when added to ReplayBuffer
    level: Optional[int] = None
    seq_len: Optional[int] = None  # Transformer sequence length
    split: Optional[str] = None


@dataclass
class SalientExperience(Experience):
    # def __init__(self):
    #     # For looking at intantiations, comment out!
    #     pass

    # The reason all these are optional is because we
    # inherit from Experience which has optional values

    # Key frame index of the sequence that this experience represents
    below_replay_index: int = None

    patch_diff: Optional[torch.Tensor] = None

    # Cluster properties
    dist: Optional[float] = None  # Distance from the core point of the dbscan cluster
    cluster_index: Optional[int] = None

    # From cluster. NOTE: There's currently
    # no fully qualified path to the replay buffer these came from
    # So if you do a salience_resume_path, the link to the original
    # lvl 0 cluster will be gone. The below_replay_index however,
    # will still be from the same RUN_ID as the replay_index, so these
    # can be assumed to be in the same parent folder.
    below_cluster_replay_indexes: Optional[np.ndarray] = None


@dataclass
class SensoryExperience(Experience):
    """
    Note if you change these properties, you'll need to change
    the line in sample_sequential that zips them
    agent_states, actions, rewards, dones, next_agent_states,
    replay_indexes, splits = zip(*exps)
    """
    state: Optional[AgentState] = None
    action: Optional[int] = None
    reward: Optional[float] = None
    done: Optional[bool] = None
    new_state: Optional[AgentState] = None

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.values())

    def is_train(self) -> bool:
        return self.split == 'train'

    def is_test(self) -> bool:
        return self.split == 'test'
