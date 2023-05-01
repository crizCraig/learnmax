from learn_max.constants import ROOT_DIR
from learn_max.salience.detect_salience import (
    create_salience_level_from_replay_buf,
)

# create_salience_level_from_replay_buf(
#     f'{ROOT_DIR}/data/replay_buff'
#     '/d_2023-03-16_13:06:27.187917'
#     '_r-TKH0PYVO_env-MontezumaRevenge-v0'
#     '/lvl_1/train',
# )


# These were within 0.5% of the kd-tree core point distances, not 50%!
# create_salience_level_from_replay_buf(
#     '/home/a/src/learnmax/data/replay_buff'
#     '/d_2023-03-31_14:14:28.665782_r-XNE4A5O7'
#     '_env-MontezumaRevenge-v0'
#     '/lvl_1/train'
# )


create_salience_level_from_replay_buf(
    replay_buf_path=(
        '/home/a/src/learnmax/data/replay_buff'
        '/d_2023-04-02_15:01:27.059619_r-DBJDS9LR'
        '_env-MontezumaRevenge-v0'
        '/lvl_1/train'
    ),
    num_points_to_cluster=400,
    frames_in_seq=8,
    dvq_decoder_path=(
        '/home/a/src/learnmax/checkpoints/dvq_0/37yv98e9/'
        'checkpoints/epoch=8-step=85999.ckpt'
    ),
)
