import sys

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.dvq_batch_size = 32
    args.should_train_gpt = False
    args.viz_dvq_clusters_knn = '--viz_dvq_clusters_knn' in sys.argv
    args.viz_dvq = '--viz_dvq' in sys.argv
    args.compress_dvq_clusters = '--compress_dvq_clusters' in sys.argv
    if args.viz_dvq or args.viz_dvq_clusters_knn or args.compress_dvq_clusters:
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/2vinku1i/checkpoints/epoch=0-step=9999.ckpt'
        args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/1jxqw8jt/checkpoints/epoch=2-step=21999.ckpt'

    args.num_embeddings = 8192
    args.embedding_dim = 4410  # decrease to get longer sequence length (aka block size / context window size)

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
