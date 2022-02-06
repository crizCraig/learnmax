"""
Defines the full (PyTorch Lightning module) VQVAE, which incorporates an
encoder, decoder and a quantize layer in the middle for the discrete bottleneck.
"""

import os
import math
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from torch import nn, einsum
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from learn_max.data.cifar10 import CIFAR10Data
from learn_max.dvq.model.deepmind_enc_dec import DeepMindEncoder, DeepMindDecoder
from learn_max.dvq.model.openai_enc_dec import OpenAIEncoder, OpenAIDecoder
from learn_max.dvq.model.openai_enc_dec import Conv2d as PatchedConv2d
from learn_max.dvq.model.quantize import VQVAEQuantize, GumbelQuantize
from learn_max.dvq.model.loss import Normal, LogitLaplace
from learn_max import dvq
from learn_max.utils import get_batch_vars, wandb_try_log

if 'TRAIN_DVQ_ONLY' in os.environ:
    dvq_module = pl.LightningModule
else:
    dvq_module = nn.Module

# -----------------------------------------------------------------------------

class VQVAE(dvq_module):

    def __init__(self, n_hid=64, num_embeddings=512, embedding_dim=64, loss_flavor='l2',
                 input_channels=3, enc_dec_flavor='deepmind', vq_flavor='vqvae', quantize_proj=None,
                 is_single_token2=False, enable_kmeans=True):
        """
        @type n_hid: number of channels controlling the size of the model
        @type num_embeddings: vocabulary size; number of possible discrete states
        @type embedding_dim: size of the vector of the embedding of each discrete token
        @type loss_flavor: `l2` or `logit_laplace`
        @type input_channels: Typically 3 for RGB
        @type enc_dec_flavor: Deepmind VQVAE or OpenAI Dall-E dVAE
        @type vq_flavor: `vqvae` or `gumbel`
        @type enable_kmeans: whether to run kmeans alongside online clustering
        """
        super().__init__()

        self.is_single_token2 = is_single_token2
        self.enable_kmeans = enable_kmeans

        # encoder/decoder module pair
        Encoder, Decoder = {
            'deepmind': (DeepMindEncoder, DeepMindDecoder),
            'openai': (OpenAIEncoder, OpenAIDecoder),
        }[enc_dec_flavor]
        self.encoder = Encoder(input_channels=input_channels, n_hid=n_hid, input_width=32,
                               embedding_dim=embedding_dim, is_single_token2=self.is_single_token2)

        if is_single_token2:
            decoder_init = quantize_proj  # embedding_dim // self.encoder.out_width ** 2
        else:
            decoder_init = embedding_dim

        self.decoder = Decoder(encoder=self.encoder, n_init=decoder_init, n_hid=n_hid,
                               output_channels=input_channels, embedding_dim=embedding_dim,
                               is_single_token2=self.is_single_token2)

        # the quantizer module sandwiched between them, +contributes a KL(posterior || prior) loss to ELBO
        QuantizerModule = {
            'vqvae': VQVAEQuantize,
            'gumbel': GumbelQuantize,
        }[vq_flavor]
        self.quantizer = QuantizerModule(self.encoder.output_channels, num_embeddings, embedding_dim,
                                         patch_width=self.encoder.out_width, output_proj=quantize_proj,
                                         is_single_token2=self.is_single_token2, enable_kmeans=self.enable_kmeans)

        # the data reconstruction loss in the ELBO
        ReconLoss = {
            'l2': Normal,
            'logit_laplace': LogitLaplace,
            # todo: add vqgan
        }[loss_flavor]
        self.recon_loss = ReconLoss
        self.global_step = 0

    def forward(self, x):

        z = self.encoder(x)
        
        z_q_emb, z_q_flat, latent_loss, z_q_ind = self.quantizer.forward(z)  # zq 128, 64, 8, 8 vs 128, 1024
        self.quantizer.global_step = self.global_step
        if 'TRY_NON_QUANTIZED' in os.environ:
            x_hat = self.decoder.forward(z)
        else:
            x_hat = self.decoder.forward(z_q_emb)  # zq is B, Embed dim, H, W i.e. 1, 10, 21, 21

        recon_loss = self.recon_loss.nll(x, x_hat)
        quant_loss_mult = float(os.getenv('QUANT_LOSS_MULT', 1))
        dvq_loss = recon_loss + quant_loss_mult * latent_loss

        return x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind  # Return x as we do a view on it

    def decode(self, z_q_emb):
        return self.decoder(z_q_emb)

    def decode_flat(self, z_q_emb_flat, output_proj):
        """
        Takes flat embedding and decodes it into an image
        """
        in_dims = z_q_emb_flat.size()  # e.g. (1, 21, 21, 10)

        width = int(np.sqrt(in_dims[-1] // output_proj))
        z_q_emb_flat = z_q_emb_flat.reshape((np.prod(in_dims[:-1]), width, width, output_proj))

        s = len(z_q_emb_flat.size())
        z_q_emb_flat = z_q_emb_flat.permute(*list(range(s - 3)), s - 1, s - 3, s - 2)
        # P = self.quantizer.output_proj
        # W = int(np.sqrt(in_dims[-1]/P))
        # z_q_emb = z_q_emb.reshape(-1, P, W, W)
        decoded = self.decode(z_q_emb_flat)

        # Match high order input dims before image dims
        out_dims = in_dims[:-1] + decoded.size()[-3:]
        decoded = decoded.reshape(*out_dims)
        return decoded

    def set_do_kmeans(self, value):
        self.enable_kmeans = value
        self.quantizer.enable_kmeans = value

    def training_step(self, batch, batch_idx):
        x = get_batch_vars(batch, use_next=True)
        x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.forward(x)
        loss = dvq_loss  # weird lightning need to call this "loss"?
        return loss, recon_loss, latent_loss, x_hat

    def validation_step(self, batch, batch_idx):
        x = get_batch_vars(batch)
        if len(x.shape) == 5:  # B, block_size, C, H, W
            # Includes gpt block size in shape, flatten first two dimensions
            x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.forward(x)
        wandb_try_log({'dvq_val_recon_loss': recon_loss}, self.global_step)
        wandb_try_log({'dvq_val_latent_loss': latent_loss}, self.global_step)

        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(z_q_ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        wandb_try_log({'dvq_val_perplexity': perplexity}, self.global_step)
        wandb_try_log({'dvq_val_cluster_use': cluster_use}, self.global_step)
        return dvq_loss, recon_loss, latent_loss, x_hat, z_q_ind, z_q_flat

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, PatchedConv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-4},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        self.optimizer = optimizer

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model type
        parser.add_argument("--vq_flavor", type=str, default='vqvae', choices=['vqvae', 'gumbel'])
        parser.add_argument("--enc_dec_flavor", type=str, default='deepmind', choices=['deepmind', 'openai'])
        parser.add_argument("--loss_flavor", type=str, default='l2', choices=['l2', 'logit_laplace'])
        # model size
        if 'SINGLE_TOKEN' in os.environ:
            default_embedding_dim = 1024
            default_num_embeddings = 8192
        elif 'SINGLE_TOKEN2' in os.environ:
            default_embedding_dim = SINGLE_TOKEN2_EMBEDDING_DIM
            default_num_embeddings = SINGLE_TOKEN2_NUM_EMBEDDINGS
        else:
            default_embedding_dim = 64
            default_num_embeddings = 512

        parser.add_argument("--num_embeddings", type=int, default=default_num_embeddings, help="vocabulary size; number of possible discrete states")
        parser.add_argument("--embedding_dim", type=int, default=default_embedding_dim, help="size of the vector of the embedding of each discrete token")
        parser.add_argument("--n_hid", type=int, default=64, help="number of channels controlling the size of the model")
        parser.add_argument("--dvq_quantize_proj", type=int, default=None)
        return parser


# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, 150000, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantizer.temperature = t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 5000, 0.0, 5e-4, trainer.global_step)
        pl_module.quantizer.kld_scale = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 1200000, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.dvq_optimizer.param_groups:
            g['lr'] = t

def cli_main():
    # TODO: Delete this. Run VQVAE through its parent LearnMax now
    pl.seed_everything(1337)

    # -------------------------------------------------------------------------
    # arguments...
    parser = ArgumentParser()
    # training related
    parser = pl.Trainer.add_argparse_args(parser)
    # model related
    parser = VQVAE.add_model_specific_args(parser)
    # dataloader related
    parser.add_argument("--data_dir", type=str, default='/apcv/users/akarpathy/cifar10')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    # done!
    args = parser.parse_args()
    # -------------------------------------------------------------------------

    data = CIFAR10Data(args)
    model = VQVAE(n_hid=args.n_hid, num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                  loss_flavor=args.loss_flavor, vq_flavor=args.vq_flavor, quantize_proj=args.dvq_quantize_proj)

    # annealing schedules for lots of constants
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='dvq_val_recon_loss', mode='min', save_top_k=3))
    callbacks.append(DecayLR())
    if args.vq_flavor == 'gumbel':
       callbacks.extend([DecayTemperature(), RampBeta()])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps=1) # 3000000)

    trainer.fit(model, data)

if __name__ == "__main__":
    cli_main()
