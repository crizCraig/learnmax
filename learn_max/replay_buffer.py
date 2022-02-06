import glob
import os
from collections import deque
from dataclasses import dataclass
from pl_bolts.datamodules.experience_source import Experience

import torch

from learn_max.agent import AgentState
from learn_max.constants import ROOT_DIR, RUN_ID, DATE_STR
from loguru import logger as log


class ReplayBuffers:
    def __init__(self,
                 env_id,
                 short_term_mem,
                 data_dir=None,
                 frames_per_file=1000,
                 train_to_test_ratio=10,
                 max_lru_size=1000,):
        """
        Disk backed (through torch.save()) replay buffer that moves experiences off-GPU to increase size
         and get better catastrophic forgetting avoidance with
        - quick length
        - quick indexing by append index in train / test
        - train and test buffer

        @param short_term_mem: Number of recent frames to keep in memory
        @param train_to_test_ratio: Number of train to test files/blocks to store
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

        self.buffer = []  # recent in-memory experience flushes to disk when reaching frames_per_file
        self.short_term_mem = short_term_mem
        self.recent_experience = deque(maxlen=short_term_mem)
        self.episode_i = 0  # Index of episode
        self.train_to_test_ratio = train_to_test_ratio  # Episodes to train vs test on
        self.flush_i = 0  # Number of all replay buffers' flushes to disk
        root_data_dir = ROOT_DIR + '/data'
        os.makedirs(root_data_dir, exist_ok=True)
        if data_dir is None:
            data_dir = f'{root_data_dir}/replay_buff_{DATE_STR}_r-{RUN_ID}_env-{env_id}'
            os.makedirs(data_dir)
        else:
            raise NotImplementedError('Need to get append_i from last episode_end_i and fill files of buffers')

        self.data_dir = data_dir
        self.test_dir = data_dir + '/test'
        self.train_dir = data_dir + '/train'
        self.test_buf = ReplayBuffer('test', self, self.test_dir, max_lru_size=max_lru_size)
        self.train_buf = ReplayBuffer('train', self, self.train_dir, max_lru_size=max_lru_size)
        self.curr_buf = self.train_buf

    def append(self, exp):
        self.curr_buf.append(exp)
        self.recent_experience.append(exp)
        if exp.done:
            self.episode_i += 1
        if self.curr_buf.just_flushed:
            self.flush_i += 1
            if (self.flush_i % (self.train_to_test_ratio+1)) == self.train_to_test_ratio:
                self.curr_buf = self.test_buf
            else:
                self.curr_buf = self.train_buf


class ReplayBuffer:
    def __init__(self, split, replay_buffers, data_dir, max_lru_size):
        self.split = split  # train or test
        self.replay_buffers = replay_buffers
        self._flush_buf = []
        self.length = 0
        self.data_dir = data_dir
        self.files = sorted(glob.glob(self.data_dir + '/*.pt'))
        os.makedirs(self.data_dir, exist_ok=True)
        self.lru = LRU(max_size=max_lru_size)
        self.just_flushed = False

    def __len__(self):
        return self.length

    def get(self, index, length=1):
        assert index >= 0
        file_i = index // self.replay_buffers.frames_per_file
        if not (file_i < len(self.files)):
            if file_i == len(self.files):
                # Requested frames are recent and have not been flushed
                return self._flush_buf[:length], []
            raise IndexError('Experience index too large')
        block = self._load(index)
        exps = block['exps'][:length]
        blocks = [block]
        if len(exps) < length and file_i != (len(self.files) - 1):
            _exps, _blocks = self.get(index + len(exps), length - len(exps))
            exps += _exps
            blocks += _blocks
        return exps, blocks

    def append(self, exp):
        assert exp.state.append_i == -1
        assert exp.new_state.append_i == -1
        exp.state.split = 1 if self.is_test() else 0
        exp.state.length = torch.tensor(self.length-1)
        exp.new_state.length = torch.tensor(self.length-1)
        self._flush_buf.append(exp)
        if len(self._flush_buf) >= self.replay_buffers.frames_per_file:
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

    def _save(self, block):
        filename = self._get_filename(self.length, self.split)
        torch.save(block, filename)
        self.lru.add(filename, block)
        self.files.append(filename)
        return filename

    def _load(self, append_i):
        file_i = append_i // self.replay_buffers.frames_per_file
        file_append_i = (file_i + 1) * self.replay_buffers.frames_per_file - 1
        filename = self._get_filename(file_append_i, self.split)
        ret = self.lru.get(filename)
        if ret is None:
            ret = torch.load(filename)
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
        self.beg = LRUNode()
        self.end = LRUNode()
        self.beg.next = self.end
        self.end.prev = self.beg
        self.mp = {}  # Could just used OrderedMap move_to_end() / popitem()

    def add(self, key, val):
        if key not in self.mp:
            if len(self.mp) == self.max_size:
                del self.mp[self.beg.next.key]
                self._remove_from_linked_list(self.beg.next)  # Remove least recently used
            node = LRUNode()
            node.val = val
            node.key = key
            self._add_to_end_of_linked_list(node)
            self.mp[key] = node
        else:
            self.get(key)  # Make node recently used

    def get(self, key):
        if key not in self.mp:
            return None
        node = self.mp[key]
        if node.next is not self.end:
            self._remove_from_linked_list(node)
            self._add_to_end_of_linked_list(node)
        return node.val

    def _remove_from_linked_list(self, node):
        assert node is not self.end
        assert node is not self.beg
        old_prev = node.prev
        old_next = node.next
        old_prev.next = old_next
        old_next.prev = old_prev

    def _add_to_end_of_linked_list(self, node):
        assert node is not self.end
        assert node is not self.beg
        old_last = self.end.prev
        self.end.prev = node
        node.prev = old_last
        old_last.next = node
        node.next = self.end
        assert old_last is not self.end.prev
        assert node is self.end.prev
        assert self.end.next is None
        assert self.beg.prev is None


@dataclass
class LRUNode:
    prev = None
    next = None
    key = None
    val = None


def test_replay_buffers_sanity():
    log.info('Testing replay buffers')
    replay_buffers = ReplayBuffers(env_id='my_env', short_term_mem=5, frames_per_file=3, train_to_test_ratio=2,
                                   max_lru_size=2)
    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_ratio):
        replay_buffers.append(Experience(state=AgentState(), action=1, reward=1, done=False, new_state=AgentState()))

    # TODO: Assert append_i, flush_i, curr_buf.is_test()
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.train_buf.length == 6
    assert replay_buffers.test_buf.length == 0
    assert replay_buffers.flush_i == 2
    assert len(replay_buffers.train_buf.files) == 2

    replay_buffers.append(Experience(state=AgentState(), action=1, reward=1, done=True, new_state=AgentState()))

    assert replay_buffers.episode_i == 1
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.test_buf.length == 1
    assert replay_buffers.curr_buf.length == 1
    assert len(replay_buffers.train_buf.files) == 2

    first_train_lru_file = list(replay_buffers.train_buf.lru.mp.keys())[0]

    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_ratio):
        replay_buffers.append(Experience(state=AgentState(), action=1, reward=1, done=False, new_state=AgentState()))

    lru_keys = list(replay_buffers.train_buf.lru.mp.keys())
    assert [0] != first_train_lru_file, 'LRU should overflow'
    assert len(lru_keys) != replay_buffers.train_buf.files
    assert replay_buffers.flush_i == 4  # 3 train, 1 test
    assert replay_buffers.curr_buf.is_train()
    assert len(replay_buffers.train_buf._flush_buf) == 1

    exps, blocks = replay_buffers.train_buf.get(0, 5)
    assert len(exps) == 5
    assert len(blocks) == 2

    exps, blocks = replay_buffers.train_buf.get(0, 1)
    assert len(exps) == 1
    assert len(blocks) == 1
    assert replay_buffers.train_buf.lru.end.prev.key.endswith('000000000002.pt')

    caught_exception = False
    try:
        replay_buffers.train_buf.get(0, 100)
    except IndexError as e:
        assert 'Experience index too large' in str(e)
        caught_exception = True
    assert caught_exception

    replay_buffers.train_buf.get(100, 1)
    log.info('Done testing replay buffers')


# Run on import so tests stay up to date
test_replay_buffers_sanity()
