from learn_max.config import get_blank_model_args, get_blank_train_args
from learn_max.model import cli_main

train_to_test_collection_files = 10


def get_model_args():
    args = get_blank_model_args()
    args.gpt_batch_size = 16
    args.gpt_seq_len = 40
    args.viz_predict_trajectory = True
    args.should_train_gpt = True
    args.dvq_checkpoint = '/home/a/src/learnmax/epoch=3-step=30999.ckpt'

    args.should_overfit_gpt = False
    args.actions_per_batch = 32

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


def get_train_args():
    args = get_blank_train_args()
    args.default_root_dir = '/home/a/src/learnmax/.lightning'

    # Model + train args
    args.train_to_test_collection_files = train_to_test_collection_files
    return args


# os.environ['ALWAYS_SAMPLE_LATEST'] = ''

cli_main(get_model_args_fn=get_model_args, get_train_args_fn=get_train_args)
