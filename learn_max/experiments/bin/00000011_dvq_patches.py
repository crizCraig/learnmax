import sys

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.single_token2 = False
    args.dvq_batch_size = 32
    args.should_train_gpt = False
    args.viz_dvq_clusters_knn = '--viz_dvq_clusters_knn' in sys.argv
    args.viz_dvq = '--viz_dvq' in sys.argv
    args.compress_dvq_clusters = '--compress_dvq_clusters' in sys.argv
    if args.viz_dvq or args.viz_dvq_clusters_knn or args.compress_dvq_clusters:
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/2vinku1i/checkpoints/epoch=0-step=9999.ckpt'
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/1jxqw8jt/checkpoints/epoch=2-step=21999.ckpt'
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/imo4u1ea/checkpoints/epoch=6-step=69999.ckpt'  # 512 n_emb
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/2t9brnyb/checkpoints/epoch=5-step=55999.ckpt'  # 256 n_emb
        # args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/1qh8t75h/checkpoints/epoch=5-step=51999.ckpt'  # 70 emb_d
        args.dvq_checkpoint = '/home/a/src/learnmax/.lightning/learnmax-learn_max_experiments_bin/37yv98e9/checkpoints/epoch=8-step=85999.ckpt'  # 70 emb_d

    args.num_state_embeddings = 256
    args.embedding_dim = 30  # decrease to get longer sequence length (aka block size / context window size)

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
