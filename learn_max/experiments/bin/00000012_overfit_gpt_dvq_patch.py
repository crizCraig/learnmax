import os

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main

train_to_test_collection_ratio = 10


def get_model_args():
    args = get_blank_model_args()
    args.gpt_batch_size = 16
    args.gpt_block_size = 40
    args.single_token2 = False
    args.viz_predict_trajectory = True
    args.should_train_gpt = True
    args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/37yv98e9/checkpoints/epoch=8-step=85999.ckpt'
    args.num_embeddings = 256
    args.embedding_dim = 30

    # Overfit args
    args.should_overfit_gpt = True
    args.actions_per_batch = 1

    # Train args
    args.train_to_test_collection_ratio = train_to_test_collection_ratio
    return args


def get_train_args():
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'

    # Model + train args
    args.train_to_test_collection_ratio = train_to_test_collection_ratio
    return args


# Tree search not implemented for patch based, perhaps can be simplified to maximize highest saliency entropy
os.environ['RAND_ACTION'] = ''

cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
