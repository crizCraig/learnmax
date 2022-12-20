import os
import string
import sys
from datetime import datetime
import random

BLOCK_SIZE = 128  # spatial extent of the model for its context
# NEPTUNE_RUN = neptune.init(project='crizcraig/safeobjective', api_token=os.environ['NEPTUNE_CREDS'])
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATE_FMT = '%Y.%m.%d_%H:%M:%S.%f'
DATE_STR = datetime.now().strftime(DATE_FMT)
SAVE_DIR = f'{ROOT_DIR}/checkpoints'
PICKLE_DIR = f'{ROOT_DIR}/pickles/{DATE_STR}'
CHECKPOINT_NAME = f'{DATE_STR}.ckpt'
SEED = 1_414_213
DEBUGGING = sys.gettrace() is not None
RUN_ID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
WANDB_MAX_LOG_PERIOD = 100
ACC_LOG_PERIOD = 10

