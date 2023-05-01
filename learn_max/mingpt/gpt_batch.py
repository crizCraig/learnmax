from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Self, Dict, Any, Tuple

import torch

from learn_max.utils import dataclass_no_none_dict


@dataclass
class GptBatchBase:
    salience_level_ind: torch.Tensor = None
    seq_len: Optional[int] = None
    type: Optional[str] = None
    cur_above_salient: Optional[torch.Tensor] = None
    next_above_salient: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.type = type(self).__name__
        self._ensure_shape()

    def _ensure_shape(self) -> None:
        pass

    def dict(self) -> dict:
        ret = dataclass_no_none_dict(self)
        return ret

    @classmethod
    def _empty_(cls: Self.__class__) -> GptBatchBase:
        ret = cls()
        ret.salience_level_ind = torch.tensor([0])
        ret.seq_len = 0
        return ret

    def __getitem__(self, batch_idx: int) -> Dict[str, Any]:
        ret = dict(
            salience_level_ind=self.salience_level_ind[batch_idx].squeeze_(0),
            seq_len=self.seq_len
        )
        ret['type'] = type(self).__name__
        return ret

    def _reshape_single_sequence(self) -> None:
        self._reshape_single_sequence_hook()
        self.salience_level_ind = self.salience_level_ind.unsqueeze(0)

    def _reshape_multi_sequence(self, a_shp: Tuple[int]) -> None:
        self._reshape_multi_sequence_hook(a_shp)
        assert len(self.salience_level_ind.shape) == 1
        self.salience_level_ind = self.salience_level_ind.view(-1, self.seq_len)

    def num_steps(self) -> int:
        return len(self.salience_level_ind.view(-1))

    def __len__(self) -> int:
        # We always have salience_level_ind
        return len(self.salience_level_ind)

    # Hooks
    def _reshape_single_sequence_hook(self) -> None:
        pass

    def _reshape_multi_sequence_hook(self, ashp: Tuple[int]) -> None:
        pass


@dataclass
class GptSalientBatch(GptBatchBase):
    salient_cluster_ind: Optional[torch.Tensor] = None

    def __getitem__(self, batch_idx: int) -> Dict:
        ret = dict(
            salient_cluster_ind=self.salient_cluster_ind[batch_idx].squeeze_(0),
            **super().__getitem__(batch_idx),
        )
        return ret

    def _reshape_single_sequence_hook(self) -> None:
        self.salient_cluster_ind = self.salient_cluster_ind.unsqueeze(0)

    def _reshape_multi_sequence_hook(self, a_shp) -> None:
        self.salient_cluster_ind = self.salient_cluster_ind.view(-1, self.seq_len)

    def empty_(self) -> None:
        self.salient_cluster_ind = torch.tensor([0])
        super()._empty_()


@dataclass
class GptSensorBatch(GptBatchBase):
    z_q_ind: Optional[torch.Tensor] = None
    actions: Optional[torch.Tensor] = None

    def empty_(self) -> GptSensorBatch:
        super()._empty_()
        self.z_q_ind = torch.tensor([0])
        self.actions = torch.tensor([0])
        return self

    def _reshape_single_sequence_hook(self) -> None:
        self.z_q_ind = self.z_q_ind.unsqueeze(0)
        self.actions = self.actions.unsqueeze(0)

    def _reshape_multi_sequence_hook(self, a_shp: Tuple[int]) -> None:
        assert self.z_q_ind.shape[0] == a_shp[0]
        self.actions = self.actions.view(-1, self.seq_len)
        self.z_q_ind = self.z_q_ind.view(
            -1, self.seq_len, *self.z_q_ind.shape[-2:]
        )

    def _ensure_shape(self) -> None:
        if self.z_q_ind is not None:
            assert None not in (self.actions, self.salience_level_ind)
            a_shp = self.actions.shape
            if len(a_shp) == 1:
                if a_shp[0] != self.seq_len:
                    # We have multiple samples/sequences,
                    # add sequence dimension (and batch if needed)
                    self._reshape_multi_sequence(a_shp)
                else:
                    # Add batch dimension as 1 if missing
                    self._reshape_single_sequence()

    def __getitem__(self, batch_idx: int) -> dict:
        ret = dict(
            z_q_ind=self.z_q_ind[batch_idx],
            actions=self.actions[batch_idx],
            **super().__getitem__(batch_idx),
        )
        return ret
