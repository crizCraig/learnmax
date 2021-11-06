# TODO: Delete this file. We do this in viz_dvq now to avoid fake args and stuff.
# ok here we go
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from learn_max.dvq.constants import SINGLE_TOKEN2_NUM_EMBEDDINGS, SINGLE_TOKEN2_EMBEDDING_DIM
from learn_max.data.cifar10 import CIFAR10Data

class FakeArgs:
    pass
args = FakeArgs()
args.data_dir = '/fileserver-csq/cvml/datasets/cifar10'
args.batch_size = 16
args.num_workers = 0
args.loss_flavor = 'l2'

data_module = CIFAR10Data(args)
val_loader = data_module.val_dataloader()
x, y = next(iter(val_loader))

plt.imshow(x[0].permute(1, 2, 0))
# plt.imshow(x[0].permute(1, 2, 0) + 0.5)


from learn_max.dvq.vqvae import VQVAE#, DEFAULT_EMBEDDING_DIM, DEFAULT_NUM_EMBEDDINGS, DEFAULT_LOSS_FLAVOR, DEFAULT_NUM_HIDDEN
args.vq_flavor = 'vqvae'
args.enc_dec_flavor = 'deepmind'
if 'SINGLE_TOKEN' in os.environ:
    args.embedding_dim = 1024
    args.num_embeddings = 8192
elif 'SINGLE_TOKEN2' in os.environ:
    args.embedding_dim = SINGLE_TOKEN2_EMBEDDING_DIM
    args.num_embeddings = SINGLE_TOKEN2_NUM_EMBEDDINGS
else:
    args.embedding_dim = 64
    args.num_embeddings = 512
args.loss_flavor = 'l2'
args.n_hid = 64

if 'SINGLE_TOKEN' in os.environ:
    model = VQVAE.load_from_checkpoint('/home/c2/src/deep-vector-quantization/lightning_logs/version_100/checkpoints/epoch=58-step=5722.ckpt', args=args)
elif 'SINGLE_TOKEN2' in os.environ:
    model = VQVAE.load_from_checkpoint('/home/c2/src/deep-vector-quantization/lightning_logs/version_152/checkpoints/epoch=54-step=21449.ckpt', args=args)

else:
    model = VQVAE.load_from_checkpoint(
        '/home/c2/src/learnmax/learn_max/lightning_logs/version_5/checkpoints/epoch=40-step=15989.ckpt',
        args=args)


model.cuda()
x = x.cuda()

x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = model(x)


xcols = torch.cat([x, x_hat], axis=2) # side by side x_pre and xhat
xrows = torch.cat([xcols[i] for i in range(x.size(0))], axis=2)


plt.figure(figsize=(20, 5))
plt.imshow((xrows.data.cpu().permute(1, 2, 0)+0.5).clamp(0, 1))
plt.axis('off')

plt.show()

# ¯\_(ツ)_/¯
# expecting something like bottom of https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
# kinda? the raw reconstruction loss on validation data this model reports is lower


"""
Things to check

- Found regression!! Was Normalization


Things to PR to andrej

- self.quantizer = QuantizerModule(self.encoder.output_channels,) _way_ fewer input channels (3 vs 128)
- output.stide (just remove as it's not used?)
- max_epochs
- namespace import fixes
- quality fix

version_7 6/12 11:38 - No normalize, no map  --- looks better!!!
version_14 6/12 12:04 - No normalize, map
version_37
version_40 clean changes with nested class solution

Things to try new Stochastic Weight Averaging  https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/

"""

