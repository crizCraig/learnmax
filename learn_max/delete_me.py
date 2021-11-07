# import argparse
#
# import gym
# import atari_py
# import fire
# import torch
# from gym import wrappers
# from loguru import logger as log
# from matplotlib import pyplot as plt
# import pytorch_lightning as pl
# import torch.backends.cudnn
# from pytorch_lightning import seed_everything
#
# from learn_max.constants import SAVE_DIR
# from learn_max.dvq.vqvae import VQVAE
# from learn_max.model import LearnMax
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-x', '--num-epochs', type=int, default=2, help="number of epochs to train for")
#     parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size to train with")
#     parser.add_argument('-l', '--block-size', type=int, default=128,
#                         help="block size for the model (length of window of context)")
#     parser.add_argument('-g', '--num-gpus', type=int, default=1, help="number of gpus to train on")
#     parser.add_argument('-n', '--num-workers', type=int, default=0, help="number of workers for dataloading")
#     parser.add_argument('-p', '--pin-memory', type=int, default=0, help="pin memory on dataloaders?")
#     parser.add_argument('-r', '--precision', type=int, default=32, help="fp precision to use, e.g. 32/16")
#     parser.add_argument('-o', '--default_root_dir', type=str, default=SAVE_DIR,
#                         help="best model checkpoint will be written at this location")
#     args = parser.parse_args()
#     log.info(vars(args))
#
#     seed_everything(0)
#
#     torch.backends.cudnn.benchmark = True  # autotune kernels
#
#     log.info("preparing the data loaders")
#
#     model = LearnMax(gpt_vocab_size=128)
#
#     common = {'batch_size': args.batch_size, 'pin_memory': bool(args.pin_memory), 'num_workers': args.num_workers}
#     train_dataloader = learn
#     val_dataloader = DataLoader(val_dataset, shuffle=False, **common)
#
#     log.info("creating the model")
#     model = GPT(train_dataset.vocab_size, args.block_size, n_layer=8, n_head=8, n_embd=256)
#
#     log.info("preparing the learning rate schedule")
#     iter_tokens = args.batch_size * args.block_size  # number of tokens backpropped in one iteration
#     epoch_tokens = math.ceil(len(train_dataset) / args.batch_size) * iter_tokens
#     lr_decay = WarmupCosineLearningRateDecay(learning_rate=6e-4,
#                                              warmup_tokens=512 * 20,  # epoch_tokens // 2,
#                                              final_tokens=args.num_epochs * epoch_tokens)
#
#     t0 = time.time()
#     log.info("training...")
#     trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.num_epochs, gradient_clip_val=1.0, callbacks=[lr_decay],
#                          precision=args.precision, default_root_dir=args.default_root_dir)
#     trainer.fit(model, train_dataloader, val_dataloader)
#     t1 = time.time()
#     log.info("%d epochs took %fs, or %fs/epoch" % (args.num_epochs, t1 - t0, (t1 - t0) / args.num_epochs))
#
#     log.info("testing...")
#     test_dataloader = DataLoader(test_dataset, shuffle=False, **common)
#     trainer.test(test_dataloaders=test_dataloader)
#
#     log.info("sampling:")
#     # context = "anarchism originated as a term of"
#     context = "O God, O God!"
#     x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...]
#     if next(model.parameters()).is_cuda:
#         x = x.cuda()
#     y = sample(model, x, 200, temperature=1.0, sample=True, top_k=None)[0]
#     completion = ''.join([train_dataset.itos[int(i)] for i in y])
#     log.info(completion)
#
#
# if __name__ == '__main__':
#     main()
#
# def main(env_id='MontezumaRevenge-v0'):
#     # MontezumaRevenge-v0 has 'repeat_action_probability': 0.25
#     # MontezumaRevenge-v4 is nondeterministic => Whether this environment is non-deterministic even after seeding
#     log.info(atari_py.list_games())
#     env = gym.make(env_id)
#
#     # You provide the directory to write to (can be an existing
#     # directory, including one with existing data -- all monitor files
#     # will be namespaced). You can also dump to a tempdir if you'd
#     # like: tempfile.mkdtemp().
#     outdir = '/tmp/random-agent-results'
#     # env = wrappers.Monitor(env, directory=outdir, force=True)
#     env.seed(0)
#     # agent = RandomAgent(env.action_space)
#
#     episode_count = 100
#     reward = 0
#     done = False
#
#     frame_encoder = VQVAE(n_hid=64, num_embeddings=1024, embedding_dim=64, loss_flavor='l2', input_channels=3,
#                           enc_dec_flavor='deepmind', vq_flavor='vqvae')
#
#     # TODO: Set this up as a lightning module
#     # TODO: Instantiate transformer - look into using detach to separate training of two. See if you can actually join them???
#
#     # transformer =
#
#     for i in range(episode_count):
#         ob = env.reset()
#         while True:
#             # action = agent.act(ob, reward, done)
#             action = 0
#             ob, reward, done, _ = env.step(action)
#             plt.imshow(ob, interpolation='nearest')
#             plt.show()
#             if done:
#                 break
#             # Note there's no env.render() here. But the environment still can open window and
#             # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
#             # Video is not recorded every episode, see capped_cubic_video_schedule for details.
#
#     # Close the env and write monitor result info to disk
#     env.close()
#
#
# if __name__ == '__main__':
#     fire.Fire(main)
#
