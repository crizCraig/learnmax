"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""
import os

import torch
import wandb
from torch import nn, einsum
import torch.nn.functional as F
from loguru import logger as log

from scipy.cluster.vq import kmeans2

# -----------------------------------------------------------------------------
from learn_max.utils import wandb_log


class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, patch_width=None, output_proj=None, is_single_token2=False,
                 enable_kmeans=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.output_proj = embedding_dim
        self.patch_width = patch_width

        self.is_single_token2 = is_single_token2
        self.forward_iter = 0
        self.enable_kmeans = enable_kmeans

        if 'SINGLE_TOKEN' in os.environ:
            self.proj = nn.Linear(embedding_dim, embedding_dim)  # Perhaps could be removed
        else:
            if self.is_single_token2:
                # TODO: We are projecting down quite a bit from w/o single token, from 64 to 10 channels, try more channels!
                # self.output_proj = 16  #patch_width ** 2
                if output_proj is not None:
                    self.output_proj = output_proj
                else:
                    self.output_proj = embedding_dim // patch_width ** 2
            self.proj = nn.Conv2d(num_hiddens, self.output_proj, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.register_buffer('data_initialized', torch.zeros(1))

        self.data_init_buffer = []
        self.data_init_points = 0
        self.initial_point_spread = None
        self.initial_centroid_spread = None

        self.global_step = 0

        # TODO: If we train the transformer and auto-encoder jointly, consider doing weight initialization in
        #  the same way for both. Right now pytorch does the dvq, with the quantizer initialized with k-means.

    def forward(self, z):
        B, C, H, W = z.size()
        z_e = self.proj(z)  #  (B, E, H, W)  # Output proj channels = E
        z_e = z_e.permute(0, 2, 3, 1)  # make (B, H, W, E)  128, 8, 8, 64
        if self.is_single_token2:
            # Enlarge token size (embedding_dim) so that we get one image per token,
            # instead of a grid of image patch tokens

            # B,  8, 8, 64 CIFAR
            # B, 21, 21, 8 Atari
            # We want to get a batch of embeddings, so n 4096. We shouldn't project down so much.
            # Fastest thing to do would be to resize, but CIFAR is 32x32 and we start out 84x84.
            # So instead of proj going down from 64 to 8, we go 64 to 10. Then the token size is
            # 10 * 441 = 4410.
            z_e = z_e.reshape(B, self.embedding_dim)  # B * H * W, E => B, H * W * E

            flatten = z_e
        else:
            # 8192 (128*8*8), 64  and flatten out space, so (B, E, H, W) -> (B*H*W, E) - a bunch of embeddings
            flatten = z_e.reshape(-1, self.embedding_dim)

        # DeepMind def does not do this but I find I have to... ;/
        # Works just as well with one point per cluster in single token CIFAR which is somewhat sus.
        # HOWEVER, super important for single token in Montezuma
        if self.enable_kmeans and self.training and self.data_initialized.item() == 0:
            # TODO: We need to do this on the batch after performing some random actions, or just try random init.
            #  If that doesn't work, we can try youtube videos, or use randomly drawn samples from a large replay
            #  buffer to retrain.
            kmeans_points_per_cluster_init = 1 if os.getenv('QUICK_KMEANS') else 15 #  orig was 64 but i think even 1 works haha.
            # We are running this every 2000 iterations so its not really data init points
            # Also periodically running kmeans could be so effective because k-means clustering is better than
            # sgd clustering + reconstruction,
            # OR it could be that the recency of the data in self.data_init_points skews clustering towards newer values.
            # Seems like the former, but would be good to rule out the latter.
            print(f'kmeans batch {round(self.data_init_points/(self.n_embed * kmeans_points_per_cluster_init) * 100, 2)}%')
            if self.data_init_points < (self.n_embed * kmeans_points_per_cluster_init):  # Ensure enough points per cluster
                self.data_init_buffer.append(flatten)
                self.data_init_points += flatten.size(0)
            else:
                # Stack data inits into tensor
                print('running kmeans!!')  # data driven initialization for the embeddings
                init_data = torch.cat(self.data_init_buffer, dim=0)
                # rp = torch.randperm(init_data.size(0))
                kd = kmeans2(init_data.data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                # kd = kmeans2(init_data[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
                self.initial_centroid_spread = self.get_centroid_spread()
                log.info(f'initial_centroid_spread {self.initial_centroid_spread}')
                self.initial_point_spread = None # reset so we get post-kmeans
                self.data_init_buffer.clear()
                self.data_init_points = 0
                self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        if self.training and self.forward_iter % 4000 == 0:
            # Causes k-means to rerun periodically which is needed for img=>token to work
            # This actually ends up happening every 2000 updates due to validation iters or something
            # TODO: Use the global train step through a lightning hook like the learning rate
            self.data_initialized.fill_(0)

        if self.initial_centroid_spread is not None:
            self.wandb_try_log({'initial_centroid_spread': self.initial_centroid_spread}, self.global_step)

        # Extract indexes from embedding and computes distance (similar to k-means here?)
        # this is the distance that's learned by the embedding table between each input token and each centroid
        # so (flatten - embed.weight)^2 = flatten^2 - 2 * flatten @ embed.weight + embed.weight^2
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        # dimensions are (num_tokens, num_embeddings)

        _, z_q_ind = (-dist).max(1)
        wandb_log({'unique_closest_clusters': torch.unique(z_q_ind).numel()}, self.global_step)
        if self.forward_iter % 100 == 0:
            wandb_log({'centroid_spread': self.get_centroid_spread()}, self.global_step)
        # Dist between centroids
        # Avg Dist betweeen points
        if not self.is_single_token2:
            # tensor([[[371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371]]], device='cuda:0')
            z_q_ind = z_q_ind.view(B, H, W)  # (128, 8, 8)

        # vector quantization cost that trains the embedding vectors
        if 'FAKE_Z_Q_EMB' in os.environ:
            z_q_ind = 0 * z_q_ind + 3070
        z_q_emb = self.embed_code(z_q_ind)  # (B, H, W, C) (128, 8, 8, 64) OR ST2=> (B, E) (128, 4096)
        point_spread = self.get_point_spread(z_e, z_q_emb)
        wandb_log({'point_spread': point_spread}, self.global_step)
        if self.initial_point_spread is None:
            self.initial_point_spread = point_spread
            log.debug(f'initial_point_spread {self.initial_point_spread}')
        wandb_log({'initial_point_spread': self.initial_point_spread}, self.global_step)

        commitment_cost = 0.25
        latent_loss = commitment_cost * (z_q_emb.detach() - z_e).pow(2).mean() + (z_q_emb - z_e.detach()).pow(2).mean()
        latent_loss *= self.kld_scale

        z_q_flat = z_q_emb  # Do this before the "noop" as it's not exactly a noop and we input this flat vector to GPT

        # noop in forward pass, straight-through gradient estimator in backward pass
        # There do end up being small differences made to z_q_emb on forward here due to floating point stuff,
        # where 93pct_delta=1.6e-10 and max=2.4e-7 for tensors with values max=3.6, min(abs(x))=3.89e-5
        z_q_emb = z_e + (z_q_emb - z_e).detach()

        if self.is_single_token2:
            # (B, E) = (B, H*W*output_proj) => (B, H, W, output_proj)
            # (128, 4410) = (128, 21*21*10) => (128, 21, 21, 10)
            z_q_emb = z_q_emb.reshape(B, H, W, self.output_proj)

        z_q_emb = z_q_emb.permute(0, 3, 1, 2)  # stack encodings into channels again: (B, C, H, W)
        self.forward_iter += 1
        return z_q_emb, z_q_flat, latent_loss, z_q_ind

    def get_point_spread(self, z_e, z_q):
        return ((z_q - z_e) ** 2).sum(axis=1).sqrt().mean()

    def get_centroid_spread(self):
        num_distances_not_self = (self.n_embed ** 2 - self.n_embed)
        centroid_spread = torch.cdist(self.embed.weight, self.embed.weight).sum() / num_distances_not_self
        return centroid_spread

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind
