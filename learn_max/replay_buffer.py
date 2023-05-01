"""
Replay buffers for storing experiences and sampling from them

Hierarchy:
    * List of ReplayBufferSplits(), one for each salience level
    * Each ReplayBufferSplits() has Two VirtualReplayBuffer()'s,
        one for train and one for test
    * Each VirtualReplayBuffer() has a list of ReplayBuffer()'s,
        one for each file, with contiguous replay indexes across all
"""
import copy
import gc
import glob
import itertools
import math
import os
import shutil
from bisect import bisect_left
from collections import deque
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional, Iterator

import pytest
import torch

from learn_max.agent import AgentState
from learn_max.constants import (
    RUN_ID,
    DATE_STR,
    DEFAULT_MAX_LRU_SIZE,
    REPLAY_FILE_PREFIX,
    REPLAY_ROOT_DIR, LEVEL_PREFIX_STR,
)
from loguru import logger as log

from learn_max.salience.experience import (
    SalientExperience,
    SensoryExperience,
    Experience,
)
from learn_max.utils import LRU, get_rand_str

REPLAY_TRAIN_DIR = 'train'
REPLAY_TEST_DIR = 'test'

class ReplayBufferSplits:
    def __init__(
            self,
            env_id: str,
            short_term_mem_max_length: int,
            steps_per_file: int = 200,
            train_to_test_collection_files: int = 10,
            max_lru_size: int = DEFAULT_MAX_LRU_SIZE,
            overfit_to_short_term: bool = False,
            verbose: bool = True,
            salience_level: int = 0,
            replay_resume_paths: Optional[List[str]] = None,
            run_id: str = RUN_ID,
    ) -> None:
        """
        Disk backed (through torch.save()) replay buffer that moves experiences off-GPU to increase size
         and get better catastrophic forgetting avoidance with
        - quick length
        - quick indexing by append index in train / test
        - train and test buffer

        @param short_term_mem_max_length: Number of recent frames to keep in memory
        @param train_to_test_collection_files: Number of train to test files/blocks to store
        @param max_lru_size: Number of files to keep in LRU memory
        @param replay_resume_paths: List of level specific (but not split) paths to replay buffers to
         resume from that are all of the same salience level e.g.:
         '/home/a/src/learnmax/data/replay_buff/d_2023-04-02_15:01:27.059619_r-DBJDS9LR_env-MontezumaRevenge-v0/lvl_1
         NOT
         '/home/a/src/learnmax/data/replay_buff/d_2023-04-02_15:01:27.059619_r-DBJDS9LR_env-MontezumaRevenge-v0/lvl_1/train'
        """
        # TODO:
        #  Allow resuming existing buffer (need to load last replay_index)
        #  Try to save unfinished episodes on crash (nice-to-have)
        #  Increase num_workers once things are moved to disk
        #  Record loss/uncertainty of episode for active learning
        #  Allow sampling
        #  - across episodes for equal batches / prediction / GPT - need to fill in without replacement (use random shuffle of episode #"s)
        #  - within episode for determining salient events

        #  LRU cache for memory based access to recent episodes (important for visualization??)
        #  - Push out oldest, add newest (deque since we don"t need delete no linked list needed)
        #  - address by key - hash map, append_i to exp (make sure to delete from hashmap when pushing out oldest)

        self.env_id = env_id
        self.steps_per_file = steps_per_file
        self.overfit_to_short_term = overfit_to_short_term
        self.verbose = verbose
        self.salience_level = salience_level
        if verbose and overfit_to_short_term:
            log.warning('Overfitting to short term mem')

        self.short_term_mem_length = short_term_mem_max_length
        self.overfit_length = short_term_mem_max_length
        self.short_term_mem: deque = deque(maxlen=short_term_mem_max_length)
        self.episode_i = 0  # Index of episode
        self.train_to_test_collection_files = train_to_test_collection_files  # Episodes to train vs test on
        self.flush_i = 0  # Number of all replay buffers' flushes to disk
        self.total_length = 0  # Count of experiences in all buffers
        os.makedirs(REPLAY_ROOT_DIR, exist_ok=True)

        data_dir = get_replay_buffers_dir(
            replay_root_dir=REPLAY_ROOT_DIR,
            run_id=run_id,
            date_str=DATE_STR,
            env_id=env_id,
            salience_level=salience_level,
        )
        os.makedirs(data_dir)
        if replay_resume_paths is None:
            replay_resume_paths = []
        self.replay_resume_paths: List[str] = replay_resume_paths
        log.info(f'Saving replay buffer to {data_dir}')
        self.data_dir = data_dir
        self.test_dir =  f'{data_dir}/{REPLAY_TEST_DIR}'
        self.train_dir = f'{data_dir}/{REPLAY_TRAIN_DIR}'
        self.test_buf = ReplayBuffer(
            parent_splits=self,
            split='test',
            data_dir=self.test_dir,
            steps_per_file=self.steps_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
            resume_paths=[f'{p}/{REPLAY_TEST_DIR}' for p in replay_resume_paths]
        )
        self.train_buf = ReplayBuffer(
            parent_splits=self,
            split='train',
            data_dir=self.train_dir,
            steps_per_file=self.steps_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
            resume_paths=[f'{p}/{REPLAY_TRAIN_DIR}' for p in replay_resume_paths]
        )

        # TODO: Remove __len__ to make it more clear that this is the
        #  total length of train and test
        self.total_length = len(self.test_buf) + len(self.train_buf)


        if self.train_to_test_collection_files == math.inf:
            self.curr_buf = self.train_buf
        else:
            self.curr_buf = self.test_buf  # Fill test buff first

    def __len__(self) -> int:
        """Number of steps in replay buffer"""
        return self.total_length

    def is_train(self) -> bool:
        return self.curr_buf.is_train()

    def resume(self, resume_paths: List[str]) -> None:
        """
        Resume from a previous replay buffer

        @param resume_paths: List of paths to replay buffers to
        resume from sorted by time / replay_index
        """
        ...

    def get_lineage(self) -> Dict[str, List[str]]:
        return dict(
            test_lineage=self.test_buf.resume_paths,
            train_lineage=self.train_buf.resume_paths,
        )

    def is_sensory(self) -> bool:
        assert self.salience_level is not None
        assert self.salience_level >= 0
        return self.salience_level == 0

    def append(self, exp: Experience) -> None:
        if not self.overfit_to_short_term:
            self.curr_buf.append(exp)
        self.short_term_mem.append(exp)
        if self.is_sensory() and exp.done:  # type: ignore
            self.episode_i += 1
        if self.curr_buf._just_flushed:
            self.flush_i += 1
            if self.train_to_test_collection_files == math.inf:
                log.warning(f'You are not collecting test data!')
            else:
                # We fill the test_buff to start, so that we have some data,
                # which makes the pattern weird at first.
                # Say you have 3 exp's per file and train to test is 2,
                # then the pattern for the first 9 exp's would be
                # as below since the test buff gets the first 3 exp's
                # that would typically have gone in train buff in
                # cadence that it will during the rest of training.
                # Pipes below delineate files
                # buf   exp_i's                                       # buf   exp_i's
                # test  0,1,2|6,7,8     vs rest of training pattern   # test  6,7,8|
                # train 3,4,5|9,10,11                                 # train 0,1,2|3,4,5|9,10,11
                # So they are the same except for 0,1,2.
                new_train = self.flush_i % (self.train_to_test_collection_files + 1)
                if new_train == self.train_to_test_collection_files:
                    self.curr_buf = self.test_buf
                else:
                    self.curr_buf = self.train_buf
        self.total_length += 1

    def delete(self) -> None:
        delete_guardrail(self.data_dir)
        log.info(
            f'Deleting replay buffer in {self.data_dir}. '
            f'NOTE that resume paths are not deleted.'
        )
        shutil.rmtree(self.data_dir)


class ReplayBuffer:
    def __init__(
        self,
        split: str,
        data_dir: str,
        steps_per_file: int,
        env_id: str,
        salience_level: int,
        parent_splits: Optional[ReplayBufferSplits] = None,
        read_only: bool = False,
        length: int = 0,
        max_lru_size: int = DEFAULT_MAX_LRU_SIZE,
        resume_paths: Optional[List[str]] = None,  # level/split specific
    ) -> None:
        self.split = split  # train or test
        self.parent_splits = parent_splits
        self.env_id = env_id
        self.salience_level = salience_level
        self.overfit_to_short_term = False
        self.global_offset = 0  # Offset in the context of a readonly resume

        if read_only:
            assert (
                    parent_splits is None
            ), 'ReplayBuffers serialization is not implemented yet'
        else:
            assert parent_splits is not None
            assert not read_only
            self.overfit_to_short_term = parent_splits.overfit_to_short_term
        self.resume_readonly: bool = read_only
        self.data_dir = data_dir
        self.files = sorted(glob.glob(self.data_dir + '/*.pt'))
        self.length = length
        self.resume_readonly_length = 0

        # This is a list of replay buffers sorted by time,
        # where only the last one is being written.
        self.resume_readonly_bufs: List[ReplayBuffer] = []

        # There's an assumption that there are 100 or fewer buffers here
        self.resume_readonly_index_map: Dict[int, ReplayBuffer] = {}

        os.makedirs(self.data_dir, exist_ok=True)
        self.max_lru_size = max_lru_size
        self.lru = self.create_lru()

        self._just_flushed = False
        self._flush_buf: List[Experience] = []
        self._steps_per_file = steps_per_file
        self._local_length = 0


        self.resume_paths: List[str] = []
        if resume_paths:
            self.resume_paths += resume_paths
            self.resume()

        if self.length > 0:
            assert self[0].replay_index == 0
            assert self[self.length - 1].replay_index is not None
            assert self.global_offset == self.resume_readonly_length
            first_exp_i = 0
            for buf in self.resume_readonly_bufs:
                if len(buf) == 0:
                    continue
                assert buf[first_exp_i].replay_index == 0
                first_exp_i += buf.length
                assert buf.resume_readonly_bufs == []
                assert buf.resume_readonly_index_map == {}
                assert buf.resume_readonly_length == 0
                assert buf.length == buf._local_length


    def resume(self) -> None:
        """
        Resume from a previous replay buffer

        self.replay_buffer_paths: List of paths to replay buffers to
        resume from

        """
        levels = [get_salience_level_from_dir(path) for path in self.resume_paths]
        assert all(lvl == self.salience_level for lvl in levels), (
            f'Levels should be {self.salience_level}, got {levels}',
        )
        for path in self.resume_paths:
            readonly_buf = get_readonly_replay_buf(path)
            assert readonly_buf is not None
            readonly_buf.global_offset = self.length
            self.length += readonly_buf.length
            self.resume_readonly_bufs.append(readonly_buf)
            self.resume_readonly_index_map[self.length - 1] = readonly_buf

        self.resume_readonly_length = self.length
        self.global_offset = self.length

    def create_lru(self) -> LRU:
        if getattr(self, 'max_lru_size', None) is None:
            self.max_lru_size = DEFAULT_MAX_LRU_SIZE
        self.lru = LRU(max_size=self.max_lru_size)
        return self.lru

    def __len__(self) -> int:
        if self.overfit_to_short_term:
            return len(self.parent_splits.short_term_mem)  # type: ignore
        return self.length

    def __getitem__(self, index: int) -> Experience:
        if isinstance(index, slice):
            raise NotImplementedError('Need to implement in get() ')
        elif self.resume_readonly:
            return self._get(index, 1)[0]
        else:
            return self.get(index, 1)[0]

    def get(
            self, start: int, length: int = 1, device: str = 'cpu'
    ) -> List[Experience]:
        if length == 0:
            return []
        if length < 0:
            raise ValueError(f'length must be >= 0, got {length}')
        start = self.get_positive_index(start)
        if start < 0:
            log.warning(f'Tried to get a negative index, {start}, returning []')
            return []
        assert self.resume_readonly_length == self.global_offset
        if not self.resume_readonly and (0 < self.resume_readonly_length <= start):
            # No resume buffers, so simply get from the current writable ReplayBuffer
            return self._get(start, length, device)

        keys = list(self.resume_readonly_index_map.keys())
        ret = []

        iters = 0
        while length > 0:
            start_buf_i = bisect_left(a=keys, x=start)
            end_buf_i = bisect_left(a=keys, x=start + length - 1)
            bufs = self.resume_readonly_bufs
            start_buf = bufs[start_buf_i] if start_buf_i < len(bufs) else self
            exps = start_buf._get(start, length, device)
            ret += exps
            iters += 1
            if start_buf_i == end_buf_i:
                break
            else:
                # Crossing buffer boundary, try grabbing rest from next buffer
                start += len(exps)
                length -= len(exps)
        if iters > 2:
            log.warning(
                f'get() spanned {iters} buffers, you are likely '
                f'loading too much data into memory.'
            )
        return ret

    def _get(self, start: int, length: int = 1, device: str = 'cpu') -> List[Experience]:
        ret = []
        start_in_global_buf = start  # Rename for clarity when dealing with multiple buffers
        del start
        while length > 0:
            start_in_global_buf = self.get_positive_index(start_in_global_buf)
            if start_in_global_buf < 0:
                log.warning(f'Tried to _get a negative index, {start_in_global_buf}')
                break
            if self.overfit_to_short_term:
                ret += list(
                    itertools.islice(
                        self.parent_splits.short_term_mem,
                        start_in_global_buf,
                        start_in_global_buf + length,
                    )
                )
                break
            start_in_local_buf = start_in_global_buf - self.global_offset
            if not (0 <= start_in_local_buf < self.length):
                # At end of buffer
                break
            file_i = start_in_local_buf // self._steps_per_file
            block_exp_i = start_in_local_buf - file_i * self._steps_per_file

            if not (file_i < len(self.files)):
                # At end of buffer
                if file_i == len(self.files):
                    # Requested frames are recent and have not been persisted,
                    # assumed to be on device
                    exps = self._flush_buf[block_exp_i:block_exp_i + length]
                    self.exps_to_device(exps, device)
                    ret += exps
                else:
                    raise NotImplementedError(
                        'Unforeseen index error, should have returned empty list'
                    )
                break
            block, block_filename = self._load_block(start_in_local_buf)
            exp_cls = SensoryExperience if self.is_sensory() else SalientExperience

            # Migrate name (delete this after a while)
            if exp_cls is SalientExperience:
                if block['exps'] and 'below_replay_indexes' in block['exps'][0]:
                    for exp in block['exps']:
                        exp['below_cluster_replay_indexes'] = exp['below_replay_indexes']
                        del exp['below_replay_indexes']
                    torch.save(block, block_filename)

            exps = block['exps'][block_exp_i:block_exp_i + length]
            exps = [exp_cls(**exp) for exp in exps]

            # Make a copy so when lightning transfers to GPU for train_batch,
            # we don't keep a reference to that GPU mem here and
            # keep it from being garbage collected,
            # thus filling the GPU.
            exps = copy.deepcopy(exps)

            self.exps_to_device(exps, device)

            ret += exps
            length -= len(exps)
            start_in_global_buf += len(exps)

        return ret

    def get_positive_index(self, start: int) -> int:
        if start < 0:
            start = len(self) - abs(start)  # index from end with negative start
        return start

    def exps_to_device(
        self, exps: List[Experience], device: Union[str, torch.device]
    ) -> None:
        if self.is_sensory():
            # Non-sensory (SalientExperiences) assumed to be
            # already on CPU
            # TODO: Check that non-sensory exps are on CPU
            for exp in exps:
                exp.state.to_(device)  # type: ignore
                exp.new_state.to_(device)  # type: ignore

    def append(self, exp: Experience) -> None:
        assert not self.resume_readonly, 'Cannot append to read only replay buffer'
        assert exp.replay_index is None
        split = 'test' if self.is_test() else 'train'
        if self.is_sensory():
            exp.state.split = split  # type: ignore
            exp.new_state.split = split  # type: ignore
        exp.split = self.split
        exp.replay_index = self.length - self.resume_readonly_length
        self._flush_buf.append(exp)
        self.length += 1
        self._local_length += 1
        # NOTE: Not threadsafe
        if len(self._flush_buf) >= self._steps_per_file:
            self._flush()
            self._just_flushed = True
        else:
            self._just_flushed = False


    def is_test(self) -> bool:
        return self.split == 'test'

    def is_train(self) -> bool:
        return self.split == 'train'

    def is_sensory(self) -> bool:
        return self.salience_level == 0

    def _flush(self) -> None:
        assert not self.resume_readonly, 'Cannot flush to read only replay buffer'
        exps = [e.__dict__ for e in self._flush_buf]  # Could just not do this
        self._save_block(exps)
        self._save_meta()
        self._flush_buf.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _save_meta(self) -> None:
        """
        Save self without LRU and replay_buffers as they're large, are circular,
        and we save blocks to disk. NOTE: Not threadsafe
        """
        lru_tmp = self.lru
        parent_tmp = self.parent_splits
        resume_readonly_tmp = self.resume_readonly_bufs
        resume_readonly_index_map_tmp = self.resume_readonly_index_map
        resume_readonly_length_tmp = self.resume_readonly_length
        length_tmp = self.length

        self.lru = None  # type: ignore
        self.parent_splits = None  # type: ignore
        self.resume_readonly = True
        self.resume_readonly_bufs = []
        self.resume_readonly_index_map = {}
        self.resume_readonly_length = 0
        self.length = self._local_length

        torch.save(self, f'{self.data_dir}/meta_{self.split}.pt')

        self.lru = lru_tmp
        self.parent_splits = parent_tmp
        self.resume_readonly = False
        self.resume_readonly_bufs = resume_readonly_tmp
        self.resume_readonly_index_map = resume_readonly_index_map_tmp
        self.resume_readonly_length = resume_readonly_length_tmp
        self.length = length_tmp

    def _save_block(self, exps: List[Experience]) -> str:
        block = dict(
            last_append_i=self.length-1,  # Last step index
            size=len(exps),
            RUN=RUN_ID,  # Note this can be different from the buffer run id if resuming
            env_id=self.env_id,
            episode_i=self.parent_splits.episode_i,
            data_dir=self.data_dir,
            exps=exps,
        )
        assert self.global_offset == self.resume_readonly_length
        last_step = self.length - self.resume_readonly_length  # Make each buff 0-based
        filename = self._get_filename(last_step, self.split)
        torch.save(block, filename)
        # self.lru.add(filename, block  # Don't add as we want to load into CPU mem from disk without grad_fn etc...
        self.files.append(filename)
        return filename

    def _load_block(
        self, replay_index: int
    ) -> Tuple[Union[SensoryExperience, SalientExperience], str]:
        file_i = replay_index // self._steps_per_file
        last_step = (file_i + 1) * self._steps_per_file
        filename = self._get_filename(last_step, self.split)
        if not os.path.exists(filename):
            filename = self._get_filename(last_step - 1, self.split)
            assert os.path.exists(filename), f'File does not exist: {filename}'
        ret = self.lru.get(filename)
        if ret is None:
            # Map to CPU so we keep LRU files in system memory and don't fill up GPU
            ret = torch.load(filename, map_location='cpu')
            self.lru.add(filename, ret)
        return ret, filename

    def _get_filename(self, last_step: int, mode: str) -> str:
        return f'{self.data_dir}/{REPLAY_FILE_PREFIX}_{mode}_{str(int(last_step)).zfill(12)}.pt'

    @property
    def steps_per_file(self) -> int:
        return self._steps_per_file

# class VirtualReplayBuffer(ReplayBuffer):
#     """
#     A virtual replay buffer that is a view into a set of replay buffers
#     which allows resuming replay buffers from previous runs.
#
#     The list of buffers is sorted by time with the first buffer containing
#     experience with replay index 0. The replay index increases contiguously
#     through the buffers. The last buffer is the one that is currently being
#     written to.
#
#     To combine multiple buffers with the same replay indices, the indices
#     for one buffer need to be rewritten and references (like clusters) need
#     to be recreated.
#
#     We also inherit from ReplayBuffer to support writing new
#     experiences to the end of the buffer.
#
#     """
#
#     def __init__(
#         self,
#         parent_splits: ReplayBufferSplits,
#         split: str,
#         data_dir: str,
#         steps_per_file: int,
#         env_id: str,
#         salience_level: int,
#         max_lru_size: int = DEFAULT_MAX_LRU_SIZE,
#         resume_paths: Optional[List[str]] = None,  # level/split specific
#     ) -> None:
#         super().__init__(
#             split=split,
#             data_dir=data_dir,
#             steps_per_file=steps_per_file,
#             env_id=env_id,
#             salience_level=salience_level,
#             parent_splits=parent_splits,
#             max_lru_size=max_lru_size,
#         )
#
#         self.length = 0
#         self.read_only_length = 0
#
#         # This is a list of replay buffers sorted by time,
#         # where only the last one is being written.
#         self.replay_buffer_list: List[ReplayBuffer] = []
#
#         # There's an assumption that there are 100 or fewer buffers here
#         self.resume_readonly_index_map: Dict[int, ReplayBuffer] = {}
#
#         self.replay_buffer_paths: List[str] = []
#         if resume_paths:
#             self.replay_buffer_paths = resume_paths
#             self.resume()
#
#         self.replay_buffer_paths.append(data_dir)
#
#         if self.length > 0:
#             assert self[0].replay_index == 0
#             assert self[self.length - 1].replay_index == self.length - 1
#
#     def __getitem__(self, index: int) -> Experience:
#         if isinstance(index, slice):
#             raise NotImplementedError('Need to implement in get()')
#         else:
#             return self.get(index, 1)[0]
#
#     def resume(self) -> None:
#         """
#         Resume from a previous replay buffer
#
#         self.replay_buffer_paths: List of paths to replay buffers to
#         resume from sorted by time / replay_index
#
#         """
#         levels = [get_level_from_replay_dir(path) for path in self.replay_buffer_paths]
#         assert all(lvl == self.salience_level for lvl in levels), (
#             f'Levels should be {self.salience_level}, got {levels}',
#         )
#         for path in self.replay_buffer_paths:
#             readonly_buf = get_readonly_replay_buf(path)
#             assert readonly_buf is not None
#             self.length += readonly_buf.length
#             self.replay_buffer_list.append(readonly_buf)
#             self.resume_readonly_index_map[self.length - 1] = readonly_buf
#
#         self.read_only_length = self.length
#
#
#     def get(
#             self, start: int, length: int = 1, device: str = 'cpu'
#     ) -> List[Experience]:
#         if length == 0:
#             return []
#         if length < 0:
#             raise ValueError(f'length must be >= 0, got {length}')
#         start = self.get_positive_index(start)
#         if start >= self.read_only_length or len(self.resume_readonly_index_map) == 0:
#             # No resume buffers, so simply get from the current writable ReplayBuffer
#             return self._get(start, length, device)
#
#         # TODO: , three buffers, tries to go before first buf,
#         #  tries to go after last buf, tries to access empty buf, test multiple at end of buff.
#         keys = list(self.resume_readonly_index_map.keys())
#         ret = []
#
#         while length > 0:
#             #    a:0    | b:1 | c:2
#             #   99, 100,  101, 102 => length = 4, start = 99
#             start_buf_i = bisect_left(a=keys, x=start)  # 1
#             end_buf_i = bisect_left(a=keys, x=start + length - 1)  # 2
#
#             bufs = list(self.resume_readonly_index_map.values())
#
#             if start_buf_i == end_buf_i:
#                 ret += bufs[start_buf_i]._get(start, length, device)
#                 break
#             else:
#                 start_buf = bufs[start_buf_i]  # 'a'
#                 end_buf = bufs[end_buf_i]  # 'b'
#                 start_buf_length = start_buf.length - start  # 101 - 99 = 2
#                 end_buf_length = length - start_buf_length  # 3 - 2 = 1
#
#                 if end_buf_i == start_buf_i + 1:
#                     ret += start_buf._get(start, start_buf_length, device)
#                     ret += end_buf._get(start + start_buf_length, end_buf_length, device)
#                     break
#                 else:
#                     log.warning(
#                         f'get() spans more than 2 buffers, you are likely '
#                         f'loading too much data into memory.'
#                     )
#                     ret += start_buf._get(start, start_buf_length, device)
#                     start += start_buf_length  # 101
#                     length -= start_buf_length  # 4 - 2 = 2
#
#         return ret


def get_readonly_replay_buf(replay_path: str) -> ReplayBuffer:
    """
    Get a replay buffer that is read only and can be used to sample from

    We should serialize the replay buffer better so we don't have to load all the files

    @param replay_path: Path to the replay buffer

    @return: Replay buffer
    """
    path = Path(replay_path)
    filenames = sorted(list(path.glob(f'{REPLAY_FILE_PREFIX}_*.pt')))
    meta_path = list(path.glob(f'meta_*.pt'))
    if len(meta_path) > 0:
        assert len(meta_path) == 1, 'Multiple meta files found'
        ret = torch.load(meta_path[0], map_location='cpu')
        if ret.lru is None:
            ret.create_lru()
        else:
            log.warning('Unexpected: LRU already exists, not creating')

        # Migrations
        if hasattr(ret, '_frames_per_file'):
            log.warning('Frames per file is deprecated, setting to steps per file')
            ret._steps_per_file = ret._frames_per_file
            del ret._frames_per_file
        if hasattr(ret, 'train_to_test_collection_files'):
            log.warning('train_to_test_collection_files is now train_to_test_collection_files')
            ret.train_to_test_collection_files = (
                ret.train_to_test_collection_files
            )
            del ret.train_to_test_collection_files
        if hasattr(ret, 'read_only'):
            log.warning('read_only is now resume_readonly')
            ret.resume_readonly = ret.read_only
            assert ret.resume_readonly is True
            del ret.read_only
        if not hasattr(ret, 'resume_readonly_bufs'):
            ret.resume_readonly_bufs = []
        if not hasattr(ret, 'resume_readonly_index_map'):
            ret.resume_readonly_index_map = {}
        if not hasattr(ret, 'resume_readonly_length'):
            # This prop only applys to writable replay buffers
            ret.resume_readonly_length = 0
        if not hasattr(ret, '_local_length'):
            # Local length is only different than length in writable replay buffers
            ret._local_length = ret.length

        return ret
    # TODO: Remove code below once meta files are saved for replay buffers you care about
    log.warning('No meta file found, loading all files, will be extremely slow')
    exps = []
    length = 0
    episode_i = 0
    steps_per_file = None
    env_id = None
    split = path.name
    level = int(path.parent.name.split('_')[-1])
    return_exps = False   # Caution, making True will load all exps into memory
    for filename in filenames:
        block = torch.load(filename, map_location='cpu')
        size = block['size']
        if env_id is None:
            env_id = block['env_id']
        else:
            assert env_id == block['env_id'], 'All files assumed to have same env_id'
        if steps_per_file is None:
            steps_per_file = size
        else:
            assert steps_per_file == size, 'All files assumed to have same size'

        if return_exps:
            length = max(length, block['last_append_i'] + 1)
            episode_i = max(episode_i, block['episode_i'])
            exps += block['exps']
        else:
            last_file = filenames[-1]
            length = int(last_file.stem.split('_')[-1])
            last_block = torch.load(last_file, map_location='cpu')
            assert length == last_block['last_append_i'] + 1
            assert steps_per_file == last_block['size']
            break  # Infer everything from first / last file
    replay_buf: ReplayBuffer = ReplayBuffer(
        split=split,
        data_dir=replay_path,
        env_id=env_id,  # type: ignore
        steps_per_file=steps_per_file,  # type: ignore
        salience_level=level,
        read_only=True,  # We don't want to mix new and old events
        length=length,
    )
    if return_exps:
        return replay_buf, exps  # type: ignore
    return replay_buf


def resume_replay_multiple_salience_levels() -> None:
    """
    For each level in the resume path, create a ReplayBufferSplits object
    then append a readonly buffer
    and add the buffer length to the current total length in the map
    """
    ...

def get_level_from_path(path: str) -> int:
    return int(path[path.rindex(LEVEL_PREFIX_STR) + len(LEVEL_PREFIX_STR):])

def get_replay_buffers_dir(
    replay_root_dir: str,
    date_str: str,
    run_id: str,
    env_id: str,
    salience_level: int,
) -> str:
    return f'{replay_root_dir}/d_{date_str}_r-{run_id}_env-{env_id}/{LEVEL_PREFIX_STR}{salience_level}'

def get_salience_level_from_dir(replay_dir: str) -> int:
    # Get the string between LEVEL_PREFIX and the next /
    return int(replay_dir.split(LEVEL_PREFIX_STR)[1].split('/')[0])


def test_resume_readonly() -> None:
    # TODO: test multiple at end of buff.
    device, splits1 = _replay_buffer_splits_helper_flush_one_file()

    assert (
        len(splits1)
        == splits1.steps_per_file * splits1.train_to_test_collection_files
    )
    initial_length = splits1.total_length
    with pytest.raises(IndexError):
        splits1.test_buf[-splits1.test_buf.length - 1]
    with pytest.raises(IndexError):
        splits1.test_buf[splits1.test_buf.length]

    empty_splits = ReplayBufferSplits(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        run_id=get_rand_str(length=20),
    )
    assert empty_splits.total_length == 0
    with pytest.raises(IndexError):
        empty_splits.test_buf[-1]

    # Create a new ReplayBufferSplits object with the same replay_resume_paths
    # but make the step_per_file 1 so that the first append flushes to disk and
    # we can resume from the new file with splits3.
    splits2 = ReplayBufferSplits(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        steps_per_file=1,
        train_to_test_collection_files=2,
        max_lru_size=2,
        verbose=False,
        replay_resume_paths=[splits1.data_dir],
        run_id=get_rand_str(length=20),
    )

    # This appends to the test buf since we reset flush_i to 0
    # when creating a new ReplayBufferSplits and test_buf is the
    # first buf to be populated.
    splits2.append(
        SensoryExperience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=splits2.total_length,
            reward=42,
            done=False,
            new_state=AgentState(),
        )
    )

    assert splits2.total_length == initial_length + 1
    assert splits2.test_buf[0].replay_index == 0
    assert splits2.test_buf[-1].replay_index == 0
    span_2 = splits2.test_buf.get(start=2, length=2)
    assert span_2[0].replay_index == 2
    assert span_2[1].replay_index == 0  # Crossed buffer boundary


    splits3 = ReplayBufferSplits(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        steps_per_file=3,
        train_to_test_collection_files=2,
        max_lru_size=2,
        verbose=False,
        replay_resume_paths=[splits1.data_dir, splits2.data_dir],
        run_id=get_rand_str(length=20),
    )

    # This appends to the test buf since we reset flush_i to 0
    # when creating a new ReplayBufferSplits and test_buf is the
    # first buf to be populated.
    splits3.append(
        SensoryExperience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=splits2.total_length,
            reward=42,
            done=False,
            new_state=AgentState(),
        )
    )

    assert splits3.total_length == splits2.total_length + 1
    assert splits3.test_buf[0].replay_index == 0
    assert splits3.test_buf[-1].replay_index == 0  # Crossed buffer boundary
    span_3 = splits3.test_buf.get(start=-4, length=4)
    assert span_3[0].replay_index == 1
    assert span_3[-1].replay_index == 0  # Crossed buffer boundary

    splits3.append(
        SensoryExperience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=splits2.total_length,
            reward=42,
            done=False,
            new_state=AgentState(),
        )
    )
    assert splits3.test_buf.get(start=-2, length=2)[1].replay_index == 1

    splits1.delete()
    splits2.delete()
    splits3.delete()


def delete_guardrail(folder: str) -> None:
    assert len(folder) > 10 and folder.count(os.path.sep) > 4, (
        'Guard against deleting high level dirs'
    )


def test_replay_buffers_sanity() -> None:
    log.info('Testing disk-backed replay buffers')
    device, replay_buffers = _replay_buffer_splits_helper_flush_one_file()

    assert (
        replay_buffers.total_length
        == replay_buffers.steps_per_file
        * replay_buffers.train_to_test_collection_files
    )
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.train_buf.length == 3
    assert replay_buffers.test_buf.length == 3
    assert replay_buffers.flush_i == 2
    assert len(replay_buffers.train_buf.files) == 1
    assert replay_buffers.short_term_mem[0].replay_index == 1
    assert replay_buffers.short_term_mem[1].replay_index == 2
    assert replay_buffers.short_term_mem[2].replay_index == 0


    # Add one new in-memory (i.e. not flushed) experience to test
    replay_buffers.append(SensoryExperience(
        state=AgentState(state=torch.tensor(0).to(device)),
        action=replay_buffers.total_length,
        reward=1,
        done=True,
        new_state=AgentState()))

    # Ensure index overflows into in-memory buffer correctly
    assert replay_buffers.test_buf.get(3)[0].action == replay_buffers.total_length - 1 == 6
    assert replay_buffers.test_buf.get(4) == []

    assert replay_buffers.episode_i == 1
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.test_buf.length == 4
    assert replay_buffers.train_buf.length == 3
    assert replay_buffers.curr_buf.length == 4
    assert len(replay_buffers.train_buf.files) == 1

    assert list(replay_buffers.train_buf.lru.mp.keys()) == []  # Don't want to add on flush/save
    replay_buffers.train_buf.get(0, 100)  # loads LRU
    first_train_lru_file = list(replay_buffers.train_buf.lru.mp.keys())[0]

    frames_to_add = (
            replay_buffers.steps_per_file * replay_buffers.train_to_test_collection_files
    )
    for i in range(frames_to_add):
        replay_buffers.append(
            SensoryExperience(
                state=AgentState(state=torch.tensor(0).to(device)),
                action=replay_buffers.total_length,
                reward=i,
                done=False,
                new_state=AgentState(),
            )
        )

    # Ensure that previously in-memory exp was flushed and has a stable index
    assert replay_buffers.test_buf.get(3)[0].action == 6

    lru_keys = list(replay_buffers.train_buf.lru.mp.keys())
    assert len(lru_keys) != replay_buffers.train_buf.files
    assert replay_buffers.flush_i == 4  # 3 train, 1 test
    assert replay_buffers.curr_buf.is_train()
    assert len(replay_buffers.train_buf._flush_buf) == 1

    exps = replay_buffers.train_buf.get(0, 5)
    assert len(exps) == 5  # Second file has no experiences
    assert [e.action for e in exps] == [3, 4, 5, 9, 10]
    assert len(replay_buffers.train_buf.files) == 2
    assert len(replay_buffers.test_buf.files) == 2
    assert [f[-5:-3] for f in replay_buffers.train_buf.files] == ['03', '06']

    exps = replay_buffers.train_buf.get(0, 1)
    assert len(exps) == 1
    assert list(replay_buffers.train_buf.lru.mp.items())[-1][0].endswith('000000000003.pt')

    exps = replay_buffers.train_buf.get(0, 100)
    assert len(exps) == len(replay_buffers.train_buf)
    assert [e.action for e in exps] == [3, 4, 5, 9, 10, 11, 12]

    exps = replay_buffers.train_buf.get(6, 100)
    assert len(exps) == 1
    assert exps[0].action == 12

    exps = replay_buffers.train_buf.get(5, 1)
    assert len(exps) == 1
    assert exps[0].action == 11

    assert replay_buffers.train_buf.get(12, 100) == []

    exps = replay_buffers.train_buf.get(100)
    assert exps == []

    exps = replay_buffers.train_buf.get(-1)
    assert exps[0].action == 12

    for i in range(
        replay_buffers.steps_per_file
        * replay_buffers.train_to_test_collection_files
    ):
        replay_buffers.append(
            SensoryExperience(
                state=AgentState(state=torch.tensor(0).to(device)),
                action=replay_buffers.total_length,
                reward=i,
                done=False,
                new_state=AgentState(),
            )
        )

    replay_buffers.train_buf.get(-2)  # Load last file into LRU
    lru_keys = list(replay_buffers.train_buf.lru.mp.keys())
    assert lru_keys[0] != first_train_lru_file, 'LRU should overflow'
    assert len(lru_keys) != replay_buffers.train_buf.files

    # TODO: Ensure GPU mem released

    replay_buffers.delete()  # TODO: Put in tear down fn
    log.info('Done testing disk-backed replay buffers')


def _replay_buffer_splits_helper_flush_one_file() -> Tuple[str, ReplayBufferSplits]:
    replay_buffers = ReplayBufferSplits(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        steps_per_file=3,
        train_to_test_collection_files=2,
        max_lru_size=2,
        verbose=False,
    )
    # Flush one file to both train and test buffers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.steps_per_file * replay_buffers.train_to_test_collection_files):
        replay_buffers.append(SensoryExperience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))
        if replay_buffers.train_buf._just_flushed:
            test_serialize(replay_buffers.train_buf)
    return device, replay_buffers


def test_serialize(expected: ReplayBuffer) -> None:
    read_only_buf = get_readonly_replay_buf(expected.data_dir)
    assert read_only_buf.resume_readonly
    assert read_only_buf.split == expected.split
    assert read_only_buf.data_dir == expected.data_dir
    assert read_only_buf.env_id == expected.env_id
    assert read_only_buf._steps_per_file == expected._steps_per_file
    assert read_only_buf.salience_level == expected.salience_level
    assert read_only_buf.length == expected.length


def test_replay_buffers_overfit() -> None:
    log.info('Testing replay buffer overfit')
    replay_buffers = ReplayBufferSplits(
        env_id='my_test_env_overfit',
        short_term_mem_max_length=5,
        steps_per_file=3,
        train_to_test_collection_files=2,
        max_lru_size=2,
        overfit_to_short_term=True,
        verbose=False,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.short_term_mem_length):
        replay_buffers.append(SensoryExperience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))

    assert replay_buffers.train_buf.files == []
    assert replay_buffers.test_buf.files == []
    exps = replay_buffers.train_buf.get(1, 3)
    assert exps[0].action == 1
    assert exps[-1].action == 3
    assert len(exps) == 3

    replay_buffers.delete()  # TODO: Put in tear down fn

    # check first, last element, and count, files=[]
    log.info('Done testing replay buffer overfit')


# Run on import so tests stay up to date
test_resume_readonly()
test_replay_buffers_overfit()
test_replay_buffers_sanity()
