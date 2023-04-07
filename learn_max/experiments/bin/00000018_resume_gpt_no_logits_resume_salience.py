import argparse
import os

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main
from learn_max.salience.salience import SalientCluster  # TODO: Move to pickle import module as these accumulate

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.gpt_batch_size = 7  # sequences per sensor batch
    args.gpt_seq_len = 8  # 10 gives OOM
    args.single_token2 = False
    args.viz_predict_trajectory = True
    args.should_train_gpt = True
    args.dvq_checkpoint = (
        '/home/a/src/learnmax/checkpoints/dvq_0/37yv98e9/'
        'checkpoints/epoch=8-step=85999.ckpt'
    )

    # args.salience_resume_path = (
    #     '/home/a/src/learnmax/pickles'
    #     '/2023-03-29_14:08:22.437325_T4JAJIKL'
    #     '/salience_store/lvl_0/2023-03-29_15:40:45.359329'
    # )  # 352MB, 1015 clusters (min_samples=3)

    # args.salience_resume_path = (
    #     '/home/a/src/learnmax/pickles'
    #     '/2023-03-30_15:26:04.393342_2F7OZYZ1/salience_store/lvl_0'
    #     '/2023-03-30_16:01:13.167669'
    # )  # folder 423.5 MB, 1562 clusters (min_samples=2)

    args.salience_resume_path = (
        '/home/a/src/learnmax/pickles'
        '/2023-04-02_13:06:01.754975_3YO0B3BX/salience_store/lvl_0'
        '/2023-04-02_13:38:53.271122'
    )  # folder 423.7 MB, 1562 clusters (min_samples=2)

    args.steps_per_abstract_replay_buff_file = 1

    args.salience_use_logits = False
    args.num_state_embeddings = 256  # 6 actions plus 1 delim => 263
    args.sensor_embedding_dim = 30
    args.salient_embedding_dim = 256

    # Overfit args
    args.should_overfit_gpt = False
    args.actions_per_batch = 8  # Higher => larger dataset, less re-sampling

    # Train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


def get_train_args() -> argparse.Namespace:
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


# Tree search not implemented for patch based, so do random actions,
# perhaps can be simplified to maximize highest saliency entropy (one step lookahead only)
os.environ['RAND_ACTION'] = ''

cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
