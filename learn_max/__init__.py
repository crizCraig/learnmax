import os
import sys

from loguru import logger as log

from learn_max.constants import ROOT_DIR, RUN_ID
from learn_max.utils import get_date_str

log.remove()
log.add(sys.stderr, level='INFO')
os.makedirs(f'{ROOT_DIR}/logs', exist_ok=True)
log.add(
    f'{ROOT_DIR}/logs/learnmax_log_{RUN_ID}_{get_date_str()}.log',
    compression='zip',
    rotation='10 MB',
)
