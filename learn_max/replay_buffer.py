import copy
import gc
import glob
import itertools
import os
import shutil
from collections import deque, OrderedDict
from dataclasses import dataclass
from pl_bolts.datamodules.experience_source import Experience

import torch

from learn_max.agent import AgentState
from learn_max.constants import ROOT_DIR, RUN_ID, DATE_STR
from loguru import logger as log


class ReplayBuffers:
    def __init__(self,
                 env_id,
                 short_term_mem_length,
                 data_dir=None,
                 frames_per_file=200,
                 train_to_test_collection_ratio=10,
                 max_lru_size=100,
                 overfit_to_short_term=False,
                 verbose=True, ):
        """
        Disk backed (through torch.save()) replay buffer that moves experiences off-GPU to increase size
         and get better catastrophic forgetting avoidance with
        - quick length
        - quick indexing by append index in train / test
        - train and test buffer

        @param short_term_mem_length: Number of recent frames to keep in memory
        @param train_to_test_collection_ratio: Number of train to test files/blocks to store
        @param max_lru_size: Number of files to keep in LRU memory
        """
        # TODO:
        #  Allow resuming existing buffer (need to load last append_i)
        #  Try to save unfinished episodes on crash (nice-to-have)
        #  Increase num_workers once things are moved to disk
        #  Record loss/uncertainty of episode for active learning
        #  Allow sampling
        #  - across episodes for equal batches / prediction / GPT - need to fill in without replacement (use random shuffle of episode #'s)
        #  - within episode for determining salient events
        #  LRU cache for memory based access to recent episodes (important for visualization??)
        #  - Push out oldest, add newest (deque since we don't need delete no linked list needed)
        #  - address by key - hash map, append_i to exp (make sure to delete from hashmap when pushing out oldest)

        self.env_id = env_id
        self.frames_per_file = frames_per_file
        self.overfit_to_short_term = overfit_to_short_term
        self.verbose = verbose
        if verbose and overfit_to_short_term:
            log.warning('Overfitting to short term mem')

        self.short_term_mem_length = short_term_mem_length
        self.short_term_mem = deque(maxlen=short_term_mem_length)
        self.episode_i = 0  # Index of episode
        self.train_to_test_collection_ratio = train_to_test_collection_ratio  # Episodes to train vs test on
        self.flush_i = 0  # Number of all replay buffers' flushes to disk
        self.total_length = 0  # Count of experiences in all buffers
        root_data_dir = ROOT_DIR + '/data'
        os.makedirs(root_data_dir, exist_ok=True)
        if data_dir is None:
            data_dir = f'{root_data_dir}/replay_buff_{DATE_STR}_r-{RUN_ID}_env-{env_id}'
            os.makedirs(data_dir)
        else:
            raise NotImplementedError('Need to get append_i from last episode_end_i and fill files of buffers')
        log.info(f'Saving replay buffer to {data_dir}')
        self.data_dir = data_dir
        self.test_dir = data_dir + '/test'
        self.train_dir = data_dir + '/train'
        self.test_buf = ReplayBuffer('test', self, self.test_dir, max_lru_size=max_lru_size)
        self.train_buf = ReplayBuffer('train', self, self.train_dir, max_lru_size=max_lru_size)
        self.curr_buf = self.test_buf  # Fill test buff first

    def append(self, exp):
        self.short_term_mem.append(exp)
        if not self.overfit_to_short_term:
            self.curr_buf.append(exp)

        if exp.done:
            self.episode_i += 1
        if self.curr_buf.just_flushed:
            self.flush_i += 1
            # We fill the test_buff to start, so that we have some data, which makes the pattern weird at first.
            # Say you have 3 exp's per file and train to test is 2, then the pattern for the first 9 exp's would be
            # as below since the test buff gets the first 3 exp's that would typically have gone in train buff in
            # cadence that it will during the rest of training. Pipes below delineate files
            # buf   exp_i's                                       # buf   exp_i's
            # test  0,1,2|6,7,8     vs rest of training pattern   # test  6,7,8|
            # train 3,4,5|9,10,11                                 # train 0,1,2|3,4,5|9,10,11
            # So they are the same except for 0,1,2.

            if (self.flush_i % (self.train_to_test_collection_ratio + 1)) == self.train_to_test_collection_ratio:
                self.curr_buf = self.test_buf
            else:
                self.curr_buf = self.train_buf
        self.total_length += 1

    def delete(self):
        log.info(f'Deleting replay buffer in {self.data_dir}')
        shutil.rmtree(self.data_dir)


class ReplayBuffer:
    def __init__(self, split, replay_buffers, data_dir, max_lru_size):
        self.split = split  # train or test
        self.replay_buffers = replay_buffers
        self.frames_per_file = replay_buffers.frames_per_file
        self.overfit_to_short_term = replay_buffers.overfit_to_short_term
        self._flush_buf = []
        self.length = 0
        self.data_dir = data_dir
        self.files = sorted(glob.glob(self.data_dir + '/*.pt'))
        os.makedirs(self.data_dir, exist_ok=True)
        self.lru = LRU(max_size=max_lru_size)
        self.just_flushed = False

    def __len__(self):
        if self.overfit_to_short_term:
            return len(self.replay_buffers.short_term_mem)
        return self.length

    def get(self, index, length=1, device='cpu'):
        if index < 0:
            index = len(self) - abs(index)
        if self.overfit_to_short_term:
            return list(itertools.islice(self.replay_buffers.short_term_mem, index, index+length))
        if not (0 <= index < self.length):
            return []
        file_i = index // self.frames_per_file
        k = index - file_i * self.frames_per_file
        if not (file_i < len(self.files)):
            if file_i == len(self.files):
                # Requested frames are recent and have not been flushed,
                # assumed to be on device
                exps = self._flush_buf[k:k+length]
                self.exps_to_device(exps, device)
                return exps
            raise NotImplementedError('Unforeseen index error, should have returned empty list')
        block = self._load(index)
        exps = block['exps'][k:k+length]
        exps = [Experience(**exp) for exp in exps]

        # Make a copy so when lightning transfers to GPU for train_batch,
        # we don't keep a reference to that GPU mem here and keep it from being garbage collected,
        # thus filling the GPU.
        exps = copy.deepcopy(exps)

        self.exps_to_device(exps, device)

        if len(exps) < length:
            exps += self.get(index + len(exps), length - len(exps), device)
        return exps

    @staticmethod
    def exps_to_device(exps, device):
        for exp in exps:
            exp.state.to(device)
            exp.new_state.to(device)

    def append(self, exp):
        assert exp.state.append_i == -1
        assert exp.new_state.append_i == -1
        exp.state.split = 1 if self.is_test() else 0
        exp.state.length = torch.tensor(self.length-1)
        exp.new_state.length = torch.tensor(self.length-1)
        self._flush_buf.append(exp)
        if len(self._flush_buf) >= self.frames_per_file:
            self._flush()
            self.just_flushed = True
        else:
            self.just_flushed = False
        self.length += 1

    def is_test(self):
        return self.split == 'test'

    def is_train(self):
        return self.split == 'train'

    def _flush(self):
        exps = [e._asdict() for e in self._flush_buf]
        block = dict(
            last_append_i=self.length-1,
            size=len(exps),
            RUN=RUN_ID,  # Note this can be different from the buffer run id if resuming
            env_id=self.replay_buffers.env_id,
            episode_i=self.replay_buffers.episode_i,
            data_dir=self.data_dir,
            exps=exps,
        )
        self._save(block)
        self._flush_buf.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def _save(self, block):
        filename = self._get_filename(self.length, self.split)
        torch.save(block, filename)
        # self.lru.add(filename, block  # Don't add as we want to load into CPU mem from disk without grad_fn etc...
        self.files.append(filename)
        return filename

    def _load(self, append_i):
        file_i = append_i // self.replay_buffers.frames_per_file
        file_append_i = (file_i + 1) * self.replay_buffers.frames_per_file - 1
        filename = self._get_filename(file_append_i, self.split)
        ret = self.lru.get(filename)
        if ret is None:
            # Map to CPU so we keep LRU files in system memory and don't fill up GPU
            ret = torch.load(filename, map_location='cpu')
            self.lru.add(filename, ret)
        return ret

    def _get_filename(self, append_i, mode):
        return f'{self.data_dir}/replay_buffer_{mode}_{str(append_i).zfill(12)}.pt'

    def _cache(self, episode):
        # TODO: if already in linked list, delete and add to end
        # TODO: Add to end of linked list and to map
        # TODO: Delete from beginning of linked list (and map) if full
        pass


class LRU:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.mp = OrderedDict()

    def add(self, key, val):
        if key not in self.mp:
            if len(self.mp) == self.max_size:
                self.mp.popitem(last=False)  # Remove least recently used
            self.mp[key] = val  # Insert at end
        else:
            self.mp.move_to_end(key)  # Make node recently used

    def get(self, key):
        if key not in self.mp:
            return None
        self.mp.move_to_end(key)
        return self.mp[key]


def test_replay_buffers_sanity():
    log.info('Testing disk-backed replay buffers')
    replay_buffers = ReplayBuffers(env_id='my_test_env', short_term_mem_length=5, frames_per_file=3,
                                   train_to_test_collection_ratio=2,
                                   max_lru_size=2, verbose=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(Experience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))

    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.train_buf.length == 3
    assert replay_buffers.test_buf.length == 3
    assert replay_buffers.flush_i == 2
    assert len(replay_buffers.train_buf.files) == 1

    replay_buffers.append(Experience(
        state=AgentState(state=torch.tensor(0).to(device)),
        action=replay_buffers.total_length,
        reward=1,
        done=True,
        new_state=AgentState()))

    assert replay_buffers.episode_i == 1
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.test_buf.length == 4
    assert replay_buffers.train_buf.length == 3
    assert replay_buffers.curr_buf.length == 4
    assert len(replay_buffers.train_buf.files) == 1

    assert list(replay_buffers.train_buf.lru.mp.keys()) == []  # Don't want to add on flush/save
    replay_buffers.train_buf.get(0, 100)  # loads LRU
    first_train_lru_file = list(replay_buffers.train_buf.lru.mp.keys())[0]

    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(Experience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))

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
    assert [f[-5:-3] for f in replay_buffers.train_buf.files] == ['02', '05']

    exps = replay_buffers.train_buf.get(0, 1)
    assert len(exps) == 1
    assert list(replay_buffers.train_buf.lru.mp.items())[-1][0].endswith('000000000002.pt')

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

    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(Experience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))

    exps = replay_buffers.train_buf.get(-2)  # Load last file into LRU
    lru_keys = list(replay_buffers.train_buf.lru.mp.keys())
    assert lru_keys[0] != first_train_lru_file, 'LRU should overflow'
    assert len(lru_keys) != replay_buffers.train_buf.files

    # TODO: Ensure GPU mem released

    replay_buffers.delete()  # TODO: Put in tear down fn
    log.info('Done testing disk-backed replay buffers')


def test_replay_buffers_overfit():
    log.info('Testing replay buffer overfit')
    replay_buffers = ReplayBuffers(env_id='my_test_env_overfit',
                                   short_term_mem_length=5,
                                   frames_per_file=3,
                                   train_to_test_collection_ratio=2,
                                   max_lru_size=2,
                                   overfit_to_short_term=True,
                                   verbose=False, )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.short_term_mem_length):
        replay_buffers.append(Experience(
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
test_replay_buffers_overfit()
test_replay_buffers_sanity()
