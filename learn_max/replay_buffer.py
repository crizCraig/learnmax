import copy
import gc
import glob
import itertools
import math
import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union, Tuple

import torch

from learn_max.agent import AgentState
from learn_max.constants import ROOT_DIR, RUN_ID, DATE_STR, DEFAULT_MAX_LRU_SIZE, REPLAY_FILE_PREFIX
from loguru import logger as log

from learn_max.utils import LRU


@dataclass
class Experience:
    replay_index: Optional[int] = None  # Set when added to ReplayBuffer
    level: Optional[int] = None
    seq_len: Optional[int] = None  # Transformer sequence length
    split: Optional[str] = None


@dataclass
class SalientExperience(Experience):
    # Key frame index of the sequence that this experience represents
    below_replay_index: Optional[int] = None

    patch_diff: Optional[torch.Tensor] = None

    # Cluster properties
    dist: Optional[float] = None  # Distance from the core point of the dbscan cluster
    cluster_index: Optional[int] = None

    # From cluster. NOTE: There's currently
    # no fully qualified path to the replay buffer these came from
    # So if you do a salience_resume_path, the link to the original
    # lvl 0 cluster will be gone. The below_repaly_index however,
    # will still be from the same run as the replay_index, so these
    # can be assumed to be in the same parent folder.
    below_cluster_replay_indexes: Optional[int] = None


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


class ReplayBuffers:
    def __init__(
            self,
            env_id: str,
            short_term_mem_max_length: int,
            data_dir: Optional[str] = None,
            steps_per_file: int = 200,
            train_to_test_collection_ratio: int = 10,
            max_lru_size: int = DEFAULT_MAX_LRU_SIZE,
            overfit_to_short_term: bool = False,
            verbose: bool = True,
            salience_level: int = 0
    ) -> None:
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
        self.frames_per_file = steps_per_file
        self.overfit_to_short_term = overfit_to_short_term
        self.verbose = verbose
        self.salience_level = salience_level
        if verbose and overfit_to_short_term:
            log.warning('Overfitting to short term mem')

        self.short_term_mem_length = short_term_mem_max_length
        self.overfit_length = short_term_mem_max_length
        self.short_term_mem: deque = deque(maxlen=short_term_mem_max_length)
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
            steps_per_file=self.frames_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
        )
        self.train_buf = ReplayBuffer(
            split='train',
            replay_buffers=self,
            data_dir=self.train_dir,
            steps_per_file=self.frames_per_file,
            env_id=self.env_id,
            salience_level=self.salience_level,
            max_lru_size=max_lru_size,
        )
        if self.train_to_test_collection_ratio == math.inf:
            self.curr_buf = self.train_buf
        else:
            self.curr_buf = self.test_buf  # Fill test buff first

    def __len__(self) -> int:
        """Number of steps in replay buffer"""
        return self.total_length

    def is_train(self) -> bool:
        return self.curr_buf.is_train()

    def is_sensory(self) -> bool:
        assert self.salience_level is not None
        assert self.salience_level >= 0
        return self.salience_level == 0

    def append(self, exp) -> None:
        if not self.overfit_to_short_term:
            self.curr_buf.append(exp)
        self.short_term_mem.append(exp)
        if self.is_sensory() and exp.done:
            self.episode_i += 1
        if self.curr_buf._just_flushed:
            self.flush_i += 1
            if self.train_to_test_collection_ratio == math.inf:
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
        split: str,
        data_dir: str,
        steps_per_file: int,
        env_id: str,
        salience_level: int,
        replay_buffers: ReplayBuffers = None,
        read_only: bool = False,
        length: int = 0,
        max_lru_size: int = DEFAULT_MAX_LRU_SIZE,
    ) -> None:
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
        self._flush_buf: List[Union[SalientExperience, SensoryExperience]] = []
        self._steps_per_file = steps_per_file



    def create_lru(self) -> LRU:
        if getattr(self, 'max_lru_size', None) is None:
            self.max_lru_size = DEFAULT_MAX_LRU_SIZE
        self.lru = LRU(max_size=self.max_lru_size)
        return self.lru

    def __len__(self):
        if self.overfit_to_short_term:
            return len(self.replay_buffers.short_term_mem)
        return self.length

    def stream(self, start_i: int = 0) -> Iterator[Union[SensoryExperience, SalientExperience]]:
        i = start_i
        while i < len(self):
            yield self.get(i)[0]
            i += 1

    def get(
        self, start: int, length: int = 1, device: str = 'cpu'
    ) -> List[Union[SensoryExperience, SalientExperience]]:
        if start < 0:
            start = len(self) - abs(start)  # index from end with negative start
        if self.overfit_to_short_term:
            return list(itertools.islice(self.replay_buffers.short_term_mem, start, start + length))
        if not (0 <= start < self.length):
            return []
        file_i = start // self._steps_per_file
        k = start - file_i * self._steps_per_file
        if not (file_i < len(self.files)):
            if file_i == len(self.files):
                # Requested frames are recent and have not been persisted,
                # assumed to be on device
                exps = self._flush_buf[k:k+length]
                self.exps_to_device(exps, device)
                return exps
            raise NotImplementedError('Unforeseen index error, should have returned empty list')
        block, block_filename = self._load_block(start)
        exp_cls = SensoryExperience if self.is_sensory() else SalientExperience

        # Migrate name (delete this after a while)
        if exp_cls is SalientExperience:
            if block['exps'] and 'below_replay_indexes' in block['exps'][0]:
                for exp in block['exps']:
                    exp['below_cluster_replay_indexes'] = exp['below_replay_indexes']
                    del exp['below_replay_indexes']
                torch.save(block, block_filename)

        exps = block['exps'][k:k+length]
        exps = [exp_cls(**exp) for exp in exps]

        # Make a copy so when lightning transfers to GPU for train_batch,
        # we don't keep a reference to that GPU mem here and keep it from being garbage collected,
        # thus filling the GPU.
        exps = copy.deepcopy(exps)

        self.exps_to_device(exps, device)

        if len(exps) < length:
            exps += self.get(start + len(exps), length - len(exps), device)
        return exps

    def exps_to_device(self, exps, device) -> None:
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
        if len(self._flush_buf) >= self._steps_per_file:
            self._flush()
            self._just_flushed = True
        else:
            self._just_flushed = False


    def is_test(self):
        return self.split == 'test'

    def is_train(self) -> bool:
        return self.split == 'train'

    def is_sensory(self) -> bool:
        return self.salience_level == 0

    def _flush(self) -> None:
        # TODO: Consider calling flush => on_disk instead as experiences
        #  are still in memory
        assert not self.read_only, 'Cannot flush to read only replay buffer'
        exps = [e.__dict__ for e in self._flush_buf]  # Could just not do this
        self._save_block(exps)
        self.save_meta()
        self._flush_buf.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def save_meta(self) -> None:
        """
        Save self without LRU and replay_buffers as they're large, are circular,
        and we save blocks to disk. NOTE: Not threadsafe
        """
        lru_save = self.lru
        parent_tmp = self.replay_buffers
        self.lru = None  # type: ignore
        self.replay_buffers = None  # type: ignore
        self.read_only = True
        torch.save(self, f'{self.data_dir}/meta_{self.split}.pt')
        self.lru = lru_save
        self.replay_buffers = parent_tmp
        self.read_only = False

    def _save_block(self, exps) -> str:
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

    def _load_block(
        self, replay_index: int
    ) -> Tuple[Union[SensoryExperience, SalientExperience], str]:
        file_i = replay_index // self._steps_per_file
        last_step = (file_i + 1) * self._steps_per_file
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
        return ret, filename

    def _get_filename(self, last_step: int, mode: str) -> str:
        return f'{self.data_dir}/{REPLAY_FILE_PREFIX}_{mode}_{str(int(last_step)).zfill(12)}.pt'

    @property
    def steps_per_file(self) -> int:
        return self._steps_per_file


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
        replay_buffer = torch.load(meta_path[0], map_location='cpu')
        if replay_buffer.lru is None:
            replay_buffer.create_lru()
        else:
            log.warning('Unexpected: LRU already exists, not creating')
        if hasattr(replay_buffer, '_frames_per_file'):
            log.warning('Frames per file is deprecated, setting to steps per file')
            replay_buffer._steps_per_file = replay_buffer._frames_per_file
        return replay_buffer
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

def test_replay_buffers_sanity() -> None:
    log.info('Testing disk-backed replay buffers')
    replay_buffers = ReplayBuffers(
        env_id='my_test_env',
        short_term_mem_max_length=5,
        steps_per_file=3,
        train_to_test_collection_ratio=2,
        max_lru_size=2,
        verbose=False,
    )
    # Flush one file to both train and test buffers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(SensoryExperience(
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

    for i in range(replay_buffers.frames_per_file * replay_buffers.train_to_test_collection_ratio):
        replay_buffers.append(SensoryExperience(
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


def test_serialize(expected):
    read_only_buf = get_readonly_replay_buf(expected.data_dir)
    assert read_only_buf.read_only
    assert read_only_buf.split == expected.split
    assert read_only_buf.data_dir == expected.data_dir
    assert read_only_buf.env_id == expected.env_id
    assert read_only_buf._steps_per_file == expected._steps_per_file
    assert read_only_buf.salience_level == expected.salience_level
    assert read_only_buf.length == expected.length


def test_replay_buffers_overfit() -> None:
    log.info('Testing replay buffer overfit')
    replay_buffers = ReplayBuffers(
        env_id='my_test_env_overfit',
        short_term_mem_max_length=5,
        steps_per_file=3,
        train_to_test_collection_ratio=2,
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
test_replay_buffers_overfit()
test_replay_buffers_sanity()
