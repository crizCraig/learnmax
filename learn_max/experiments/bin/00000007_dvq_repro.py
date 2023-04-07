import os

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.dvq_batch_size = 32
    args.should_train_gpt = False

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


def get_train_args():
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
