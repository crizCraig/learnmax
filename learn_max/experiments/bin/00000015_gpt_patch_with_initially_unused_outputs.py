import os

from learn_max.config import get_blank_model_args, get_blank_train_args
os.environ['DISABLE_SALIENCE_TESTS'] = ''
from learn_max.model import cli_main
from learn_max.salience.salience import SalientCluster  # TODO: Move to pickle import module as these accumulate

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.gpt_batch_size = 7
    args.gpt_seq_len = 8  # 10 gives OOM
    args.single_token2 = False
    args.viz_predict_trajectory = True
    args.should_train_gpt = True
    args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/37yv98e9/checkpoints/epoch=8-step=85999.ckpt'

    args.salience_resume_path = '/home/a/src/learnmax/pickles/2022.11.30_14:14:45.578047'
    args.num_state_embeddings = 256  # 6 actions plus 1 delim => 263
    args.embedding_dim = 30

    # Overfit args
    args.should_overfit_gpt = False
    args.actions_per_batch = 8  # Higher => larger dataset, less re-sampling

    # Train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


def get_train_args():
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


# Tree search not implemented for patch based, perhaps can be simplified to maximize highest saliency entropy
os.environ['RAND_ACTION'] = ''

cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
