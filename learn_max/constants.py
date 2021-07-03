import os
import sys
from datetime import datetime

import neptune.new as neptune

BLOCK_SIZE = 128  # spatial extent of the model for its context
NEPTUNE_RUN = neptune.init(project='crizcraig/safeobjective', api_token=os.environ['NEPTUNE_CREDS'])
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATE_STR = datetime.now().strftime('%Y_%m-%d_%H-%M.%S.%f')
SAVE_DIR = f'{ROOT_DIR}/checkpoints'
CHECKPOINT_NAME = f'{DATE_STR}.ckpt'
IS_DEBUG_MODE = getattr(sys, 'gettrace', None)


