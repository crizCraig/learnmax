import argparse
import os

import pytorch_lightning as pl

from learn_max.constants import DEBUGGING
from loguru import logger as log


def get_blank_model_args():
    args = get_model_args_from_cli()
    _check_model_defaults(args)
    return args


def get_blank_train_args():
    args = get_train_args_from_cli()
    _check_train_defaults(args)
    return args


def get_model_args_from_cli():
    parser = argparse.ArgumentParser()
    # model args
    parser = add_reinforcement_learning_args(parser)
    parser = add_model_specific_args(parser)
    parser = add_shared_model_train_args(parser)
    args, unknown = parser.parse_known_args()
    if args.num_workers is None:
        # data loader workers - pycharm has issues debugging when > 0
        # also weirdness when >0 in that wandb.init needs to be called for quantize to log???
        #   - must be due to spawning multiple training processes?
        args.num_workers = 0 if DEBUGGING else 0
        print('cli num workers', args.num_workers)
        print('DEBUGGING', DEBUGGING)
    return args


def get_train_args_from_cli():
    """
    Note there should be no overlap with model args, esp if setting args in python as args you set
     in get_model_args will not 'stick' and you'll need to set them again in train args since
     we don't pass args through global command line, i.e. sys.argv, when setting in python e.g. in experiments/ python
     files.
    """
    # common = {'batch_size': args.gpt_batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
    # trainer args  # TODO: Check that our defaults above are preserved for overlapping things like pin-memory
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--num_epochs', type=int, default=1000, help="number of epochs to train for")
    parser.add_argument("--num_gpus", type=int, default=1)  # TODO: Use lightning's num_modes arg here?
    parser = add_shared_model_train_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    return args


def add_reinforcement_learning_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
    """
    Adds arguments for DQN model

    Note:
        These params are fine tuned for Pong env.

    Args:
        arg_parser: parent parser
    """
    arg_parser.add_argument(
        "--warm_start_size",
        type=int,
        default=int(os.getenv('WARM_START', 10_000)),
        help="how many samples do we use to fill our buffer at the start of training",
    )

    arg_parser.add_argument("--batches_per_epoch", type=int, default=10_000, help="number of batches per pseudo (RL) epoch")
    arg_parser.add_argument("--env_id", type=str, help="gym environment tag", default='MontezumaRevenge-v0')
    arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    arg_parser.add_argument(
        "--avg_reward_len",
        type=int,
        default=100,
        help="how many episodes to include in avg reward",
    )

    return arg_parser


def add_model_specific_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
    arg_parser.add_argument("--dvq_quantize_proj", type=int, default=10)
    arg_parser.add_argument('--num_workers', type=int, default=None, help="number of workers for dataloading")
    arg_parser.add_argument("--single_token2", action='store_true', default=True)
    arg_parser.add_argument('--viz_dvq', action='store_true', help="visualize dvq images", default=False)
    arg_parser.add_argument('--viz_all_dvq_clusters', action='store_true', help="visualize all dvq clusters", default=False)
    arg_parser.add_argument('--viz_dvq_clusters_knn', action='store_true', help="visualize dvq clusters knn", default=False)
    arg_parser.add_argument('--compress_dvq_clusters', action='store_true', help="compress dvq clusters", default=False)
    arg_parser.add_argument('--viz_predict_trajectory', action='store_true', help="visualize all dvq clusters", default=False)
    arg_parser.add_argument('--dvq_checkpoint', type=str, help="DVQ checkpoint to restore", default=None)
    arg_parser.add_argument('--checkpoint', type=str, help="Full (DVQ+GPT) checkpoint to restore", default=None)
    arg_parser.add_argument('--should_train_gpt', action='store_true', help="Whether to train GPT", default=False)
    arg_parser.add_argument('--gpt_learning_rate', type=float, help="GPT batch size", default=6e-4)
    arg_parser.add_argument('--gpt_batch_size', type=int, help="GPT batch size", default=8)
    arg_parser.add_argument('--gpt_seq_len', type=int, help="sequence length for the model (length of temporal window)", default=40)
    arg_parser.add_argument('--actions_per_batch', type=int, help="avoids overfitting with more data generated between updates", default=1)
    return arg_parser


def add_shared_model_train_args(arg_parser: argparse.ArgumentParser, ) -> argparse.ArgumentParser:
    # TODO: Use pin_memory in dataloader when using disk-backed replay buffer
    log.warning('Not pinning memory, do this once testing overfitting!')
    arg_parser.add_argument('-p', '--pin_memory', type=bool, default=False, help="pin memory on dataloaders?")
    arg_parser.add_argument('--train_to_test_collection_ratio', type=float, help="num train to test examples to collect in replay buffer", default=10)
    return arg_parser


def _check_model_defaults(args):
    # Assert some defaults are set, TODO: Use constants for these
    assert args.checkpoint is None
    assert args.dvq_checkpoint is None
    assert args.gpt_batch_size == 8
    assert args.num_workers == 0


def _check_train_defaults(args):
    # Assert some defaults are set, TODO: Use constants for these
    assert args.num_gpus == 1
    assert args.num_epochs == 1000

