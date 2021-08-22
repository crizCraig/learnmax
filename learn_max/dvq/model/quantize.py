"""
The critical quantization layers that we sandwich in the middle of the autoencoder
(between the encoder and decoder) that force the representation through a categorical
variable bottleneck and use various tricks (softening / straight-through estimators)
to backpropagate through the sampling process.
"""
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2

# -----------------------------------------------------------------------------

class VQVAEQuantize(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, patch_width=None, output_proj=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.kld_scale = 10.0

        self.output_proj = embedding_dim
        self.patch_width = patch_width

        if 'SINGLE_TOKEN' in os.environ:
            self.proj = nn.Linear(embedding_dim, embedding_dim)  # Perhaps could be removed
        else:
            if 'SINGLE_TOKEN2' in os.environ:
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

        # TODO: If we train the transformer and auto-encoder jointly, consider doing weight initialization in
        #  the same way for both. Right now pytorch does the dvq, with the quantizer initialized with k-means.

    def forward(self, z, wait_to_init):
        if 'SINGLE_TOKEN' in os.environ:
            B, E = z.size()  # B, Embed dim
            z_e = self.proj(z)
            flatten = z_e
        else:
            B, C, H, W = z.size()
            z_e = self.proj(z)  #  (B, E, H, W)  # Output proj channels = E
            z_e = z_e.permute(0, 2, 3, 1)  # make (B, H, W, E)  128, 8, 8, 64
            if 'SINGLE_TOKEN2' in os.environ:
                # Enlarge token size (embedding_dim) so that we get one image per token,
                # instead of a grid of image patch tokens

                # 128, 8, 8, 64 CIFAR
                # B, 21, 21, 8 Atari
                # We want to get a batch of embeddings, so n 4096. We shouldn't project down so much, number 1.
                # Fastest thing to do would be to resize, but CIFAR is 32x32 and we start out 84x84.
                # So instead of proj going down from 64 to 8, we should go 64 to 10. Then the token size can be
                # 10 * 441 = 4410.
                z_e = z_e.reshape(B, self.embedding_dim)  # B * H * W, E => B, H * W * E

                flatten = z_e
            else:
                flatten = z_e.reshape(-1, self.embedding_dim)  # 8192 (128*8*8), 64  and flatten out space, so (B, E, H, W) -> (B*H*W, E) - a bunch of embeddings

        # DeepMind def does not do this but I find I have to... ;/
        # Works just as well with one point per cluster in single token regime which is somewhat sus.
        if False and not wait_to_init and self.training and self.data_initialized.item() == 0:
            # TODO: We need to do this on the batch after performing some random actions, or just try random init.
            #  If that doesn't work, we can try youtube videos, or use randomly drawn samples from a large replay
            #  buffer to retrain.
            kmeans_points_per_cluster_init = 1 if os.getenv('QUICK_KMEANS') else 32 #  orig was 64 but i think even 1 works haha.
            print(f'kmeans batch {round(self.data_init_points/(self.n_embed * kmeans_points_per_cluster_init) * 100)}%')
            if self.data_init_points < self.n_embed * kmeans_points_per_cluster_init:  # Ensure enough points per cluster
                self.data_init_buffer.append(flatten)
                self.data_init_points += flatten.size(0)
            else:
                # Stack data inits into tensor
                print('running kmeans!!') # data driven initialization for the embeddings
                init_data = torch.cat(self.data_init_buffer, dim=0)
                # rp = torch.randperm(init_data.size(0))
                kd = kmeans2(init_data.data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                # kd = kmeans2(init_data[rp[:20000]].data.cpu().numpy(), self.n_embed, minit='points')  # flatten: 512,1024 vs 8192,64
                self.embed.weight.data.copy_(torch.from_numpy(kd[0]))
                self.data_init_buffer.clear()
                self.data_initialized.fill_(1)
            # TODO: this won't work in multi-GPU setups

        # Extract indexes from embedding and computes distance (similar to k-means here?)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, ind = (-dist).max(1)
        if 'SINGLE_TOKEN' not in os.environ and 'SINGLE_TOKEN2' not in os.environ:
            # tensor([[[371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371],
            #          [371, 371, 371,  ..., 371, 371, 371]]], device='cuda:0')
            ind = ind.view(B, H, W)  # (128, 8, 8)
        # Single token initial 128
        # tensor([411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411, 411,
        #         411, 411], device='cuda:0')

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C) (128, 8, 8, 64) OR ST2=> (B, E) (128, 4096)
        commitment_cost = 0.25
        latent_loss = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        latent_loss *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        if 'SINGLE_TOKEN2' in os.environ:
            # Had 128 * 64 = B * W **2, E
            # Now we have B, W ** 2 * C = 128,
            z_q = z_q.reshape(B, H, W, self.output_proj)  # (B, E) = (B, H*W*C) => (B, H, W, C)
        if 'SINGLE_TOKEN' not in os.environ:
            z_q = z_q.permute(0, 3, 1, 2) # stack encodings into channels again: (B, C, H, W)

        return z_q, latent_loss, ind

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
