import os
import string
import sys
from datetime import datetime
import random

RUN_ID = ''.join(
    random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
)
BLOCK_SIZE = 128  # spatial extent of the model for its context
# NEPTUNE_RUN = neptune.init(project='crizcraig/safeobjective', api_token=os.environ['NEPTUNE_CREDS'])
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPLAY_ROOT_DIR = f'{ROOT_DIR}/data/replay_buff'
DATE_FMT = '%Y-%m-%d_%H:%M:%S.%f'
DATE_STR = datetime.now().strftime(DATE_FMT)
SAVE_DIR = f'{ROOT_DIR}/checkpoints'
PICKLE_DIR = f'{ROOT_DIR}/pickles/{DATE_STR}_{RUN_ID}'
CHECKPOINT_NAME = f'{DATE_STR}.ckpt'
SEED = 1_414_213
DEBUGGING = sys.gettrace() is not None
WANDB_MAX_LOG_PERIOD = 100
ACC_LOG_PERIOD = 10

# Human would be ~10k? and zuma like 10
MAX_NUM_SALIENCE_LEVELS = 10

# Human would be ~1M? and zuma like 10k
MAX_SALIENT_CARDINALITY = 10_000

DEFAULT_MAX_LRU_SIZE = 100
REPLAY_FILE_PREFIX = 'replay_buffer'
NUM_DIFF_SALIENT = 2
COMBINE_STEPS_ABSTRACT_SEQ = 1
COMBINE_STEPS_SENSOR_SEQ = 8
TRAIN = 'train'
TEST = 'test'
DEFAULT_GPT_SEQ_LEN = 8
LEVEL_PREFIX_STR = 'lvl_'
