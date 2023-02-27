import copy
import gc
import glob
import itertools
import os
import shutil
from collections import deque, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learn_max.agent import AgentState
from learn_max.constants import ROOT_DIR, RUN_ID, DATE_STR, DEFAULT_MAX_LRU_SIZE, REPLAY_FILE_PREFIX
from loguru import logger as log


@dataclass
class SalientExperience:
    replay_index: int = None
    below_replay_index: int = None
    patch_diff: np.ndarray = None
    seq_len: int = None
    dist: float = None
    split: str = None
    level: int = None
    cluster_index: int = None

    # From cluster. NOTE: There's currently
    # no fully qualified path to the replay buffer these came from
    # So if you do a salience_resume_path, the link to the original
    # lvl 0 cluster will be gone. The below_repaly_index however,
    # will still be from the same run as the replay_index, so these
    # can be assumed to be in the same parent folder.
    below_replay_indexes: int = None


@dataclass
class Experience:
    """
    Note if you change these properties, you'll need to change the line in sample_sequential that zips them
    agent_states, actions, rewards, dones, next_agent_states, replay_indexes, splits = zip(*exps)
    """
    state: AgentState
    action: int
    reward: float
    done: bool
    new_state: AgentState
    replay_index: int = None
    split: str = None

    def __iter__(self):
        return iter(self.__dict__.values())

    def is_train(self):
        return self.split == 'train'

    def is_test(self):
        return self.split == 'test'


class ReplayBuffers:
    def __init__(
            self,
            env_id,
            short_term_mem_max_length,
            data_dir=None,
            frames_per_file=200,
            train_to_test_collection_ratio=10,
            max_lru_size=DEFAULT_MAX_LRU_SIZE,
            overfit_to_short_term=False,
            verbose=True,
            salience_level=0
    ):
        """
        Disk backed (through torch.save()) replay buffer that moves experiences off-GPU to increase size
         and get better catastrophic forgetting avoidance with
        - quick length
        - quick indexing by append index in train / test
        - train and test buffer

        @param short_term_mem_max_length: Number of recent frames to keep in memory
        @param train_to_test_collection_ratio: Number of train to test files/blocks to store
        @param max_lru_size: Number of files to keep in LRU memory
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
        self.frames_per_file = frames_per_file
        self.overfit_to_short_term = overfit_to_short_term
        self.verbose = verbose
        self.salience_level = salience_level
        if verbose and overfit_to_short_term:
            log.warning('Overfitting to short term mem')

        self.short_term_mem_length = short_term_mem_max_length
        self.overfit_length = short_term_mem_max_length
        self.short_term_mem = deque(maxlen=short_term_mem_max_length)
        self.episode_i = 0  # Index of episode
        self.train_to_test_collection_ratio = train_to_test_collection_ratio  # Episodes to train vs test on
        self.flush_i = 0  # Number of all replay buffers' flushes to disk
        self.total_length = 0  # Count of experiences in all buffers
        root_data_dir = f'{ROOT_DIR}/data/replay_buff'
        os.makedirs(root_data_dir, exist_ok=True)
        if data_dir is None:
            data_dir = f'{root_data_dir}/d_{DATE_STR}_r-{RUN_ID}_env-{env_id}/lvl_{salience_level}'
            os.makedirs(data_dir)
        else:
            # Also note that resuming should deal with branching. So we should copy the
            # old replay buffer or deal with logical branching and a virtual replay
            # index space to map to original buffer.
            raise NotImplementedError('Need to get replay_index from last episode_end_i and fill files of buffers')
        log.info(f'Saving replay buffer to {data_dir}')
        self.data_dir = data_dir
        self.test_dir = data_dir + '/test'
        self.train_dir = data_dir + '/train'
        self.test_buf = ReplayBuffer(
            split='test',
            replay_buffers=self,
            data_dir=self.test_dir,
            frames_per_file=self.frames_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
        )
        self.train_buf = ReplayBuffer(
            split='train',
            replay_buffers=self,
            data_dir=self.train_dir,
            frames_per_file=self.frames_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
        )
        self.curr_buf = self.test_buf  # Fill test buff first

    def is_train(self):
        return self.curr_buf.is_train()

    def is_sensory(self):
        assert self.salience_level is not None
        assert self.salience_level >= 0
        return self.salience_level == 0

    def append(self, exp):
        if not self.overfit_to_short_term:
            self.curr_buf.append(exp)
        self.short_term_mem.append(exp)
        if self.is_sensory() and exp.done:
            self.episode_i += 1
        if self.curr_buf._just_flushed:
            self.flush_i += 1
            # We fill the test_buff to start, so that we have some data, which makes the pattern weird at first.
            # Say you have 3 exp's per file and train to test is 2, then the pattern for the first 9 exp's would be
            # as below since the test buff gets the first 3 exp's that would typically have gone in train buff in
            # cadence that it will during the rest of training. Pipes below delineate files
            # buf   exp_i's                                       # buf   exp_i's
            # test  0,1,2|6,7,8     vs rest of training pattern   # test  6,7,8|
            # train 3,4,5|9,10,11                                 # train 0,1,2|3,4,5|9,10,11
            # So they are the same except for 0,1,2.
            new_train = self.flush_i % (self.train_to_test_collection_ratio + 1)
            if new_train == self.train_to_test_collection_ratio:
                self.curr_buf = self.test_buf
            else:
                self.curr_buf = self.train_buf
        self.total_length += 1

    def delete(self):
        log.info(f'Deleting replay buffer in {self.data_dir}')
        shutil.rmtree(self.data_dir)


class ReplayBuffer:
    def __init__(
        self,
        split,
        data_dir,
        frames_per_file,
        env_id,
        salience_level,
        replay_buffers=None,
        read_only=False,
        length=0,
        max_lru_size=DEFAULT_MAX_LRU_SIZE,
    ):
        self.split = split  # train or test
        self.replay_buffers = replay_buffers
        self.env_id = env_id
        self.salience_level = salience_level
        self.overfit_to_short_term = False

        if read_only:
            assert (
                replay_buffers is None
            ), 'ReplayBuffers serialization is not implemented yet'
        else:
            assert replay_buffers is not None
            assert not read_only
            self.overfit_to_short_term = replay_buffers.overfit_to_short_term
        self.read_only = read_only
        self.data_dir = data_dir
        self.files = sorted(glob.glob(self.data_dir + '/*.pt'))
        self.length = length
        os.makedirs(self.data_dir, exist_ok=True)
        self.max_lru_size = max_lru_size
        self.lru = self.create_lru()

        self._just_flushed = False
        self._flush_buf = []
        self._frames_per_file = frames_per_file

    def create_lru(self):
        if getattr(self, 'max_lru_size', None) is None:
            self.max_lru_size = DEFAULT_MAX_LRU_SIZE
        self.lru = LRU(max_size=self.max_lru_size)
        return self.lru

    def __len__(self):
        if self.overfit_to_short_term:
            return len(self.replay_buffers.short_term_mem)
        return self.length

    def get(self, start, length=1, device='cpu'):
        if start < 0:
            start = len(self) - abs(start)  # index from end with negative start
        if self.overfit_to_short_term:
            return list(itertools.islice(self.replay_buffers.short_term_mem, start, start + length))
        if not (0 <= start < self.length):
            return []
        file_i = start // self._frames_per_file
        k = start - file_i * self._frames_per_file
        if not (file_i < len(self.files)):
            if file_i == len(self.files):
                # Requested frames are recent and have not been persisted,
                # assumed to be on device
                exps = self._flush_buf[k:k+length]
                self.exps_to_device(exps, device)
                return exps
            raise NotImplementedError('Unforeseen index error, should have returned empty list')
        block = self._load(start)
        exps = block['exps'][k:k+length]
        exp_cls = Experience if self.is_sensory() else SalientExperience

        exps = [exp_cls(**exp) for exp in exps]

        # Make a copy so when lightning transfers to GPU for train_batch,
        # we don't keep a reference to that GPU mem here and keep it from being garbage collected,
        # thus filling the GPU.
        exps = copy.deepcopy(exps)

        self.exps_to_device(exps, device)

        if len(exps) < length:
            exps += self.get(start + len(exps), length - len(exps), device)
        return exps

    def exps_to_device(self, exps, device):
        if self.is_sensory():
            # Non-sensory (SalientExperiences) assumed to be
            # already on CPU
            # TODO: Check that non-sensory exps are on CPU
            for exp in exps:
                exp.state.to_(device)
                exp.new_state.to_(device)

    def append(self, exp):
        assert not self.read_only, 'Cannot append to read only replay buffer'
        assert exp.replay_index is None
        split = 'test' if self.is_test() else 'train'
        if self.is_sensory():
            exp.state.split = split
            exp.new_state.split = split
        exp.split = self.split
        exp.replay_index = self.length
        self._flush_buf.append(exp)
        self.length += 1
        # NOTE: Not threadsafe
        if len(self._flush_buf) >= self._frames_per_file:
            self._flush()
            self._just_flushed = True
        else:
            self._just_flushed = False


    def is_test(self):
        return self.split == 'test'

    def is_train(self):
        return self.split == 'train'

    def is_sensory(self):
        return self.salience_level == 0

    def _flush(self):
        # TODO: Consider calling flush => on_disk instead as experiences
        #  are still in memory
        assert not self.read_only, 'Cannot flush to read only replay buffer'
        exps = [e.__dict__ for e in self._flush_buf]
        self._save_block(exps)
        self.save_meta()
        self._flush_buf.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def save_meta(self):
        """
        Save self without LRU and replay_buffers as they're large, are circular,
        and we save blocks to disk. NOTE: Not threadsafe
        """
        lru_save = self.lru
        parent_save = self.replay_buffers
        self.lru = None
        self.replay_buffers = None
        self.read_only = True
        torch.save(self, f'{self.data_dir}/meta_{self.split}.pt')
        self.lru = lru_save
        self.replay_buffers = parent_save
        self.read_only = False

    def _save_block(self, exps):
        block = dict(
            last_append_i=self.length-1,  # Last step index
            size=len(exps),
            RUN=RUN_ID,  # Note this can be different from the buffer run id if resuming
            env_id=self.env_id,
            episode_i=self.replay_buffers.episode_i,
            data_dir=self.data_dir,
            exps=exps,
        )
        filename = self._get_filename(self.length, self.split)
        torch.save(block, filename)
        # self.lru.add(filename, block  # Don't add as we want to load into CPU mem from disk without grad_fn etc...
        self.files.append(filename)
        return filename

    def _load(self, replay_index):
        file_i = replay_index // self._frames_per_file
        last_step = (file_i + 1) * self._frames_per_file
        filename = self._get_filename(last_step, self.split)
        if not os.path.exists(filename):
            log.warning(f'File does not exist: {filename}, checking for old file indexing (minus 1)')
            filename = self._get_filename(last_step - 1, self.split)
            assert os.path.exists(filename), f'File does not exist: {filename}'
            log.success('Found old file indexing, loading')
        ret = self.lru.get(filename)
        if ret is None:
            # Map to CPU so we keep LRU files in system memory and don't fill up GPU
            ret = torch.load(filename, map_location='cpu')
            self.lru.add(filename, ret)
        return ret

    def _get_filename(self, last_step, mode):
        return f'{self.data_dir}/{REPLAY_FILE_PREFIX}_{mode}_{str(int(last_step)).zfill(12)}.pt'


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

def get_readonly_replay_buf(replay_path):
    """
    Get a replay buffer that is read only and can be used to sample from

    We should serialize the replay buffer better so we don't have to load all the files

    @param replay_path: Path to the replay buffer

    @return: Replay buffer
    """
    path = Path(replay_path)
    filenames = list(path.glob(f'{REPLAY_FILE_PREFIX}_*.pt'))
    meta_path = list(path.glob(f'meta_*.pt'))
    if len(meta_path) > 0:
        assert len(meta_path) == 1, 'Multiple meta files found'
        replay_buffers = torch.load(meta_path[0], map_location='cpu')
        if replay_buffers.lru is None:
            replay_buffers.create_lru()
        else:
            log.warning('Unexpected: LRU already exists, not creating')
        return replay_buffers
    # TODO: Remove code below once meta files are saved for replay buffers you care about
    log.warning('No meta file found, loading all files, will be extremely slow')
    exps = []
    length = 0
    episode_i = 0
    frames_per_file = None
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
        if frames_per_file is None:
            frames_per_file = size
        else:
            assert frames_per_file == size, 'All files assumed to have same size'

        if return_exps:
            length = max(length, block['last_append_i'] + 1)
            episode_i = max(episode_i, block['episode_i'])
            exps += block['exps']
        else:
            last_file = sorted(filenames)[-1]
            length = int(last_file.stem.split('_')[-1])
            last_block = torch.load(last_file, map_location='cpu')
            assert length == last_block['last_append_i'] + 1
            assert frames_per_file == last_block['size']
            break  # Infer everything from first / last file
    replay_buf = ReplayBuffer(
        split=split,
        data_dir=replay_path,
        env_id=env_id,
        frames_per_file=frames_per_file,
        salience_level=level,
        read_only=True,  # We don't want to mix new and old events
        length=length,
    )
    if return_exps:
        return replay_buf, exps
    return replay_buf

def test_replay_buffers_sanity():
    log.info('Testing disk-backed replay buffers')
    replay_buffers = ReplayBuffers(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        frames_per_file=3,
        train_to_test_collection_ratio=2,
        max_lru_size=2,
        verbose=False,
    )
    # Flush one file to both train and test buffers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(Experience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))
        if replay_buffers.train_buf._just_flushed:
            test_serialize(replay_buffers.train_buf)

    assert replay_buffers.total_length == replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio
    assert replay_buffers.curr_buf.is_test()
    assert replay_buffers.train_buf.length == 3
    assert replay_buffers.test_buf.length == 3
    assert replay_buffers.flush_i == 2
    assert len(replay_buffers.train_buf.files) == 1
    assert replay_buffers.short_term_mem[0].replay_index == 1
    assert replay_buffers.short_term_mem[1].replay_index == 2
    assert replay_buffers.short_term_mem[2].replay_index == 0


    # Add one new in-memory (i.e. not flushed) experience to test
    replay_buffers.append(Experience(
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

    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(Experience(
            state=AgentState(state=torch.tensor(0).to(device)),
            action=replay_buffers.total_length,
            reward=i,
            done=False,
            new_state=AgentState()))

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
        replay_buffers.frames_per_file
        * replay_buffers.train_to_test_collection_ratio
    ):
        replay_buffers.append(
            Experience(
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


def test_serialize(expected):
    read_only_buf = get_readonly_replay_buf(expected.data_dir)
    assert read_only_buf.read_only
    assert read_only_buf.split == expected.split
    assert read_only_buf.data_dir == expected.data_dir
    assert read_only_buf.env_id == expected.env_id
    assert read_only_buf._frames_per_file == expected._frames_per_file
    assert read_only_buf.salience_level == expected.salience_level
    assert read_only_buf.length == expected.length


def test_replay_buffers_overfit():
    log.info('Testing replay buffer overfit')
    replay_buffers = ReplayBuffers(env_id='my_test_env_overfit',
                                   short_term_mem_max_length=5,
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
