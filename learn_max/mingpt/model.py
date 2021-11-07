"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import wandb
from loguru import logger as log
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention but I am including an
    explicit implementation to show that there is nothing too scary here.
    """

    def __init__(self, embedding_dim, block_size, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert embedding_dim % n_head == 0
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # 64, 128, 256, Channels = token+pos embedding size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(  B, T, self.n_head, C // self.n_head).transpose(1, 2) # (64, 8, 128, 32)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,  nh, T,  hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,  nh, T,  hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_dim, block_size, n_head, attn_pdrop, resid_pdrop, out_embedding_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, block_size, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(resid_pdrop),  # TODO: Ask Eleuther why this expansion and contraction is done even in perceiver
        )
        if out_embedding_dim != embedding_dim:
            self.out_proj = nn.Linear(embedding_dim, out_embedding_dim)
        else:
            self.out_proj = None

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if self.out_proj is not None:
            # We could do cross attention as in perceiver IO, but this seems simpler ¯\_(ツ)_/¯
            x = self.out_proj(x)
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self,
                 # model definition args
                 vocab_size: int,  # size of the vocabulary (number of possible tokens)
                 block_size: int,  # length of the model's context window in time
                 n_layer: int,  # depth of the model; number of Transformer blocks in sequence
                 embedding_dim: int,  # the "width" of the model, number of channels in each Transformer
                 n_head: int,  # number of heads in each multi-head attention inside each Transformer block
                 # model optimization args
                 learning_rate: float = 3e-4,  # the base learning rate of the model
                 weight_decay: float = 0.1,  # amount of regularizing L2 weight decay on MatMul ops
                 betas: Tuple[float, float] = (0.9, 0.95),  # momentum terms (betas) for the Adam optimizer
                 embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
                 resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
                 attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix
                 should_input_embed: bool = False,  # whether to use embeddings or indexes as input to first layer
                 ):
        super().__init__()
        self.vocab_size = vocab_size

        # save these for optimizer init later
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        # input embedding stem: drop(content + position)
        self.tok_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
        self.drop = nn.Dropout(embd_pdrop)
        # deep transformer: just a sequence of transformer blocks
        self.blocks, _, out_dims = self.init_blocks(attn_pdrop, block_size, embedding_dim, n_head, n_layer, resid_pdrop)
        out_embedding_dim = out_dims[-1]

        # decoder: at the end one more layernorm and decode the answers
        self.ln_f = nn.LayerNorm(out_embedding_dim)
        self.logit_p_head = nn.Linear(out_embedding_dim, vocab_size, bias=False) # no need for extra bias due to one in ln_f
        self.deviation_head = nn.Linear(out_embedding_dim, vocab_size, bias=False)   # mean deviation

        self.block_size = block_size
        self.apply(self._init_weights)

        self.iter = 0
        self.trajectory_counts: Dict[Tuple[int], int] = defaultdict(int)
        self.max_trajectory_count = 0
        self.should_input_embed = should_input_embed

        log.info("number of parameters: %e" % sum(p.numel() for p in self.parameters()))

    def init_blocks(self, attn_pdrop, block_size, embedding_dim, n_head, n_layer, resid_pdrop):
        """
        Here we allow different in/out neuron counts in each transformer layer aka block. Transformers usually
        keep the same number of neurons at every layer, but a la Perceiver IO, these types of things can be changed
        like any MLP. This was originally motivated here by OOM where reducing the output of the first layer
        reduces the amount of memory used in all subsequent layers that use the reduced layer width.
        """
        blocks = []
        approx_scale = [0.1] + [1] * (n_layer - 1)  # TODO: Allow passing this in
        assert len(approx_scale) == n_layer
        out_dims = []
        for l_i in range(n_layer):
            embedding_dim = self.make_divisible_by_heads(embedding_dim, n_head)
            out_embedding_dim = self.make_divisible_by_heads(int(approx_scale[l_i] * embedding_dim), n_head)
            blocks.append(Block(embedding_dim, block_size, n_head, attn_pdrop, resid_pdrop, out_embedding_dim))
            embedding_dim = out_embedding_dim
            out_dims.append(out_embedding_dim)
        return nn.Sequential(*blocks), approx_scale, out_dims

    def make_divisible_by_heads(self, embedding_dim, n_head):
        if embedding_dim % n_head != 0:
            # Make embedding dim divisible by number of heads
            embedding_dim -= embedding_dim % n_head
        return embedding_dim

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        """
        Vanilla model initialization:
        - all MatMul weights \in N(0, 0.02) and biases to zero
        - all LayerNorm post-normalization scaling set to identity, so weight=1, bias=0
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas)
        self.optimizer = optimizer
        return optimizer

    def count_trajectories(self, targets):
        # 1, 3, 4, 5
        trajectories = [tuple(x) for x in targets.numpy()]
        for t in trajectories:
            t_small = t[-5:]
            self.trajectory_counts[t_small] += 1
            self.max_trajectory_count = max(self.max_trajectory_count, self.trajectory_counts[t_small])

        if self.iter % 5 == 0:
            print('trajectories: ', len(self.trajectory_counts))
            print('max_trajectory_count: ', self.max_trajectory_count)

    def forward(self, idx_or_embed):
        if len(idx_or_embed.size()) == 3:  # TODO: Use self.should_input_embed here
            b, t, embed = idx_or_embed.size()
        else:
            b, t = idx_or_embed.size()
            embed = None
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        if embed is None:
            token_embeddings = self.tok_emb(idx_or_embed) # each index maps to a (learnable) vector
        else:
            token_embeddings = idx_or_embed  # Tokens are input directly from dvq, keeping semantic info instead of re-learning
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.logit_p_head(x)
        expected_deviation = self.deviation_head(x)  # Uses mean deviation instead of standard deviation https://stats.stackexchange.com/q/81986/18187

        wandb.log({'train/expected_deviation_median': torch.quantile(expected_deviation, 0.5)})
        wandb.log({'train/expected_deviation_90pct': torch.quantile(expected_deviation, 0.9 )})
        wandb.log({'train/expected_deviation_95pct': torch.quantile(expected_deviation, 0.95)})
        wandb.log({'train/expected_deviation_99pct': torch.quantile(expected_deviation, 0.99)})
        wandb.log({'train/expected_deviation_mean': expected_deviation.mean()})
        wandb.log({'train/expected_deviation_max': expected_deviation.max()})
        wandb.log({'train/expected_deviation_min': expected_deviation.min()})
        wandb.log({'train/logits_std': logits.std()})

        return logits, expected_deviation

    def step_(self, split, batch, batch_idx=None):
        embed, idx, target_idx = batch  # So this is like  x, x_idx, y = batch
        x = embed if self.should_input_embed else idx
        logits, expected_deviation = self(x)

        # Calculate mean deviation loss --------------------------------------
        # Turn targets into one hot B x block_size x vocab_size with 1 in vocab
        one_hot = F.one_hot(target_idx, num_classes=self.vocab_size).squeeze()
        probs = F.softmax(logits, dim=-1)
        wandb.log({'train/probs_std': probs.std()})
        p_diff = (one_hot - probs).abs()  # actual deviation
        d_diff = p_diff - expected_deviation
        d_loss = d_diff.square().sum() / d_diff.numel()

        # Calculate standard transformer categorical probability loss-----------
        # pytorch cross entropy has built-in softmax so pass logits
        p_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_idx.reshape(-1))
        loss = d_loss + p_loss
        self.iter += 1
        return {'loss': loss}

    def training_step(self, *args, **kwargs):
        return self.step_('train', *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.step_('val', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.step_('test', *args, **kwargs)
