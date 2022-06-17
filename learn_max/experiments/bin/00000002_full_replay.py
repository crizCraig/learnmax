import os

from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main


def get_model_args():
    args = get_blank_model_args()
    args.gpt_batch_size = 16
    args.gpt_seq_len = 40
    args.viz_predict_trajectory = True
    args.should_train_gpt = True
    args.dvq_checkpoint = '/home/a/src/learnmax/epoch=3-step=30999.ckpt'

    args.should_overfit_gpt = False
    args.actions_per_batch = 1
    return args


def get_train_args():
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'
    return args


# os.environ['ALWAYS_SAMPLE_LATEST'] = ''

cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
