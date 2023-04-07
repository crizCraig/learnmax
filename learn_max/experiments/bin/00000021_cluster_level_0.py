from learn_max.constants import ROOT_DIR
from learn_max.salience.detect_salience import (
    create_salience_level_from_replay_buf,
)

create_salience_level_from_replay_buf(
    f'{ROOT_DIR}/data/replay_buff'
    '/d_2022.12.17_17:45:07.646449'
    '_r-GFE45YIT_env-MontezumaRevenge-v0'
    '/lvl_0/train',
    frames_in_seq=8,
    state_tokens_in_frame=121,
)

# OUT!
# min_samples=3
# /home/a/src/learnmax/pickles/2023-03-29_14:08:22.437325_T4JAJIKL/salience_store/lvl_0
# level 1 clusters not being added fast enough - 1015 clusters

# min_samples = 2
# /home/a/src/learnmax/pickles/2023-03-30_15:26:04.393342_2F7OZYZ1/salience_store/lvl_0/2023-03-30_16:01:13.167669
