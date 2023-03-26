from learn_max.constants import ROOT_DIR
from learn_max.salience.detect_salience import (
    create_salience_level_from_replay_buf,
)

create_salience_level_from_replay_buf(
    f'{ROOT_DIR}/data/replay_buff'
    '/d_2022.12.17_17:45:07.646449'
    '_r-GFE45YIT_env-MontezumaRevenge-v0'
    '/lvl_0/train',
    frames_in_sequence_window=8,
)
