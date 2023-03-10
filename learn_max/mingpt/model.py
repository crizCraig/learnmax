"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import collections
import math
from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger as log
from torch.nn import functional as F

from learn_max.constants import ACC_LOG_PERIOD
from learn_max.mingpt.utils import add_non_state_tokens
from learn_max.utils import (
    accuracy,
    wandb_log,
    sa2as,
    GPT_BATCH_TYPE_MAP,
    GptSalientBatch,
)


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
        # log.debug('entered attn forward')
        B, T, C = x.size()  # 64, 128, 256, Channels = token+pos embedding size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(  B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (64, 8, 128, 32)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,  nh, T,  hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,  nh, T,  hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

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
        # log.debug('starting gpt attn')
        x = x + self.attn(self.ln1(x))
        # log.debug('done with gpt attn')
        x = x + self.mlp(self.ln2(x))
        if self.out_proj is not None:
            # We could do cross attention as in perceiver IO, but this seems simpler
            x = self.out_proj(x)
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self,
                 # model definition args
                 output_size: int,  # size of the output vocabulary
                 num_input_embeddings: int,  # number of possible input tokens
                 num_state_embeddings: int,  # number of possible state tokens (i.e. not action tokens or delimiter token)
                 frames_in_sequence_window: int,  # number of frames in sequence window (block)
                 n_layer: int,  # depth of the model; number of Transformer blocks in sequence
                 input_embedding_dim: int,  # the "width" of the input to the model
                 n_head: int,  # number of heads in each multi-head attention inside each Transformer block
                 # model optimization args
                 learning_rate: float = 3e-4,  # the base learning rate of the model
                 weight_decay: float = 0.1,  # amount of regularizing L2 weight decay on MatMul ops
                 betas: Tuple[float, float] = (0.9, 0.95),  # momentum terms (betas) for the Adam optimizer
                 embd_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on input embeddings
                 resid_pdrop: float = 0.1,  # \in [0,1]: amount of dropout in each residual connection
                 attn_pdrop: float = 0.1,  # \in [0,1]: amount of dropout on the attention matrix
                 should_input_embed: bool = False,  # whether to use embeddings or indexes as input to first layer
                 should_input_and_learn_embed: bool = False,  # whether to cat input embed with learned token embed
                 num_actions: int = 0,  # number of actions to use for action embedding
                 is_single_token2: bool = False,  # whether there's one token per image
                 state_tokens_in_frame: int = None,  # number of state tokens (i.e. not action or delimeter) in frame
                 tokens_in_frame: int = None,  # number of tokens per frame (1 in single token)
                 tokens_in_salient_step: int = None,  # number of tokens per salient step
                 batch_size: int = None,
                 ):
        super().__init__()
        self.output_size = output_size
        self.is_single_token2 = is_single_token2
        self.batch_size = batch_size
        self.steps_in_sequence_window = frames_in_sequence_window
        self.sequences_per_salient_experience = 2
        self.frames_in_salient_experience = (
                self.steps_in_sequence_window * self.sequences_per_salient_experience
        )
        assert self.frames_in_salient_experience > 0
        self.state_tokens_in_frame = state_tokens_in_frame
        self.tokens_in_frame = tokens_in_frame
        self.tokens_in_salient_step = tokens_in_salient_step

        # save these for optimizer init later
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        # input embedding stem: drop(content + position)
        #  Eventually:
        #     content = prev_salient + next_salient + dvq_token + learned_token + action_token
        # action_embedding_dim = 100
        embedding_dim = input_embedding_dim  # + action_embedding_dim
        # self.dvq_proj = nn.Linear(input_embedding_dim, embedding_dim)
        assert embedding_dim % n_head == 0, f'Embedding len {embedding_dim} not evenly divisible by number of heads {n_head}'

        # self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)

        # This should be num of z_q_emb
        self.tok_emb = nn.Embedding(num_input_embeddings, input_embedding_dim)
        self.tok_emb_salience_levels = []  # TODO: Add to this as we cluster new salience levels

        # block_size is the length of the model's context window in time
        if is_single_token2:
            tokens_in_seq = frames_in_sequence_window
        else:
            tokens_in_seq = tokens_in_frame * frames_in_sequence_window
        self.pos_emb = nn.Parameter(torch.zeros(1, tokens_in_seq, embedding_dim))
        self.patch_pos_emb = nn.Parameter(
            torch.zeros(1, tokens_in_frame, embedding_dim)
        )
        self.frame_pos_emb = nn.Parameter(
            torch.zeros(1, frames_in_sequence_window, tokens_in_frame, embedding_dim)
        )

        # Only for single token, tok_emb does actions and delim otherwise
        self.act_emb = nn.Embedding(num_actions, input_embedding_dim)
        # self.delim_emb = nn.Embedding(1, embedding_dim)

        self.drop = nn.Dropout(embd_pdrop)
        # deep transformer: just a sequence of transformer blocks
        self.blocks, _, out_dims = self.init_blocks(
            attn_pdrop, tokens_in_seq, embedding_dim, n_head, n_layer, resid_pdrop
        )
        out_embedding_dim = out_dims[-1]

        # decoder: at the end one more layernorm and decode the answers
        self.ln_f = nn.LayerNorm(out_embedding_dim)
        self.logit_p_head = nn.Linear(out_embedding_dim, output_size, bias=False)  # no need for extra bias due to one in ln_f
        self.salient_logit_heads = []  # TODO: Add to this as we cluster new salience levels

        # TODO: Remove or change to deviation over longer timescale - we want to know how the entropy of the
        #   probs changes over time so that, in the case of an aleatoric process like a slot machine, we see that
        #   while there may be patterns in recent data, the system over long stretches is random. The deviation head
        #   just predicts instantaneous changes to probability and so does not do this.
        self.deviation_head = nn.Linear(out_embedding_dim, output_size, bias=False)   # mean deviation

        self.target_idx = torch.arange(output_size)

        self.tok_in_seq = tokens_in_seq
        self.apply(self._init_weights)

        self.iter = 0
        self.trajectory_counts: Dict[Tuple[int], int] = defaultdict(int)
        self.max_trajectory_count = 0
        self.should_input_embed = should_input_embed
        self.should_input_and_learn_embed = should_input_and_learn_embed
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.num_input_embeddings = num_input_embeddings
        self.num_state_embeddings = num_state_embeddings

        self.global_step = 0

        # Track most recent maxlen batches of logits so we can normalize logits for salience detection
        logit_sum_len = 100  # TODO: consider increasing to get better normalization
        self.logit_dq_sum = collections.deque(maxlen=logit_sum_len)
        self.logit_dq_numel = collections.deque(maxlen=logit_sum_len)
        self.logit_dq_std = collections.deque(maxlen=logit_sum_len)

        log.info("number of parameters: %e" % sum(p.numel() for p in self.parameters()))

    def init_blocks(self, attn_pdrop, block_size, embedding_dim, n_head, n_layer, resid_pdrop):
        """
        Here we allow different in/out neuron counts in each transformer layer aka block. Transformers usually
        keep the same number of neurons at every layer, but a la Perceiver IO, these types of things can be changed
        like any MLP. This was originally motivated here by OOM where reducing the output of the first layer
        reduces the amount of memory used in all subsequent layers that use the reduced layer width.
        """
        blocks = []
        if self.is_single_token2:
            approx_scale = [0.1] + [1] * (n_layer - 1)  # TODO: Allow passing this in
        else:
            approx_scale = [1] * n_layer
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
        return self.tok_in_seq

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

        # special case the position embedding parameters in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('patch_pos_emb')
        no_decay.add('frame_pos_emb')

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

    def forward(self, *args):
        if self.is_single_token2:
            embed, z_q_ind, actions = args
            return self.single_token_forward(actions, embed, z_q_ind)
        else:
            z_q_ind, actions, salient_cluster_ind, salience_level_ind = args

            # TODO: Create embedding for goal token and concatenate with other tokens
            tok_ind = self.add_non_state_tokens(
                z_q_ind,
                actions,
                salient_cluster_ind,
                salience_level_ind
            )

            B, FiS, TiF = tok_ind.size()  # batch, frames in sequence, tokens in frame
            assert TiF == self.tokens_in_frame
            assert TiF * FiS <= self.tok_in_seq, 'Cannot forward, model sequence length is exhausted.'
            token_embed = self.tok_emb(tok_ind)  # each input token index maps to a (learnable) vector
            assert FiS <= self.steps_in_sequence_window
            assert TiF <= self.tokens_in_frame

            # map frame and patch positions to a learnable vector
            frame_pos_embed = self.frame_pos_emb[:, :FiS]  # support partial sequences
            patch_pos_embed = self.patch_pos_emb[:, :TiF]  # support partial frames

            x = token_embed + frame_pos_embed + patch_pos_embed  # broadcast sum embeddings
            x = x.reshape(B, TiF * FiS, self.embedding_dim)
            x = self.drop(x)
            x = self.blocks(x)
            x = self.ln_f(x)
            # Not currently used, but could be useful (YAGNI?)
            logits = self.logit_p_head(x)


            # Uses mean deviation instead of standard deviation
            # https://stats.stackexchange.com/q/81986/18187
            expected_deviation = self.deviation_head(x)

            wandb_log({'train/logits_std': logits.std()})

            return logits, expected_deviation

    def get_salience_logits(self, actions, z_q_ind, z_q_emb, replay_ind):
        """
        Get logits for z_q_ind and actions
        @param actions:
        @param z_q_ind:
        @param z_q_emb:
        @param replay_ind:

            For example, if I open a door, there are lots of new possibilities.
        @return:
        """
        _, S, H, W = z_q_ind.shape
        # Break forward up into multiple self.frames_in_sequence_window tensors
        z_q_ind_f = z_q_ind.reshape(S // self.steps_in_sequence_window, -1, H * W)
        actions_f = actions.reshape(S // self.steps_in_sequence_window, -1)
        logits, expected_deviation = self.forward(z_q_ind_f, actions_f)
        # Normalize logits - These need to be normalized by dividing by the average of recent logits,
        #   that way we can compare old saliences to new ones even when logits increase over time.
        #   Thinking the increasing logits could be from GELU activations where activations are
        #   increasingly surpassing the zero cuttoff.
        #   Since the data in a minibatch is sequential, we need to average multiple minibatches
        #   to reduce correlation and determine a more global salience.
        SHOULD_NORMALIZE = False
        if SHOULD_NORMALIZE:
            self.logit_dq_sum.append(float(logits.sum()))
            self.logit_dq_numel.append(float(logits.numel()))
            self.logit_dq_std.append(float(logits.std()))
            if len(self.logit_dq_std) < self.logit_dq_std.maxlen:
                return None  # need to bootstrap more std()
            avg_recent = sum(self.logit_dq_sum) / sum(self.logit_dq_numel)
            if len(self.logit_dq_sum) == 1:
                assert np.isclose(float(logits.mean()), avg_recent)
            ret = (logits - avg_recent) / np.mean(self.logit_dq_std)  # z-normalize
        else:
            ret = logits
        assert ret.std() < 1E10, 'Logit variance is blowing up'
        wandb_log({'salience/logits_u': logits.mean(), 'salience/logits_std': logits.std()})

        return ret   # W, (H*W+a+d) * FiS, E

    def single_token_forward(self, actions, embed, idx):
        # if self.should_input_embed:
        #     b, t, embed = idx_or_embed.size()
        # else:
        #     b, t = idx_or_embed.size()
        #     embed = None
        b, t, e = embed.size()
        assert t <= self.tok_in_seq, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        token_embed = self.tok_emb(idx)  # each input token index maps to a (learnable) vector
        position_embed = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        action_embed = self.act_emb(actions)  # each action maps to a (learnable) vector
        # token_embed = torch.cat((token_embed, action_embeddings), dim=2)
        # allow learning a transformation into the summed representation below so that we can learn how to best
        # distribute the dvq embedding to the attention heads
        # embed = self.dvq_proj(embed)  # this projection kills performance for some reason
        # x = self.drop(token_embed + position_embeddings + action_embeddings + dvq_proj)
        x = self.drop(token_embed + position_embed + action_embed)  # + embed)
        x = self.blocks(x)   # 1, 40, 4410 = B, S, H*W*dvq_proj
        x = self.ln_f(x)
        logits = self.logit_p_head(x)
        expected_deviation = self.deviation_head(
            x)  # Uses mean deviation instead of standard deviation https://stats.stackexchange.com/q/81986/18187
        # wandb.log({'train/expected_deviation_median': torch.quantile(expected_deviation, 0.5)})
        # wandb.log({'train/expected_deviation_90pct': torch.quantile(expected_deviation, 0.9 )})
        # wandb.log({'train/expected_deviation_95pct': torch.quantile(expected_deviation, 0.95)})
        # wandb.log({'train/expected_deviation_99pct': torch.quantile(expected_deviation, 0.99)})
        # wandb.log({'train/expected_deviation_mean': expected_deviation.mean()})
        # wandb.log({'train/expected_deviation_max': expected_deviation.max()})
        # wandb.log({'train/expected_deviation_min': expected_deviation.min()})
        wandb_log({'train/logits_std': logits.std()})
        return logits, expected_deviation

    def step_(self, split, batch, batch_idx=None):
        if self.is_single_token2:
            embed, z_q_ind, next_idx, a, a_next = batch  # gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y
            if split == 'train':
                assert embed.size()[1] == self.tok_in_seq, \
                    'Not filling block size in train will result in untrained latter positions.'

            target_idx = self.s_i_to_as_i(next_idx, a_next)
            logits, expected_deviation = self.forward(embed, z_q_ind, a)  # B, S
        else:
            z_q_ind = batch.z_q_ind
            actions = batch.actions
            salient_cluster_ind = (
                batch.salient_cluster_ind if isinstance(batch, GptSalientBatch) else None
            )
            salience_level_ind = batch.salience_level_ind
            B, FiS, H, W = z_q_ind.shape
            TiF = H * W
            assert TiF <= self.tokens_in_frame  # delim and action not added yet
            z_q_ind = z_q_ind.reshape(B, FiS, TiF)
            if split == 'train':
                assert FiS * TiF >= self.tok_in_seq, \
                    'Not filling block size in train will result in untrained latter positions.'
            # Shift targets by one patch
            # target_idx = z_q_ind[:,1:,:]
            target_idx = self.add_non_state_tokens(
                z_q_ind, actions, salient_cluster_ind, salience_level_ind
            )
            assert target_idx.shape[-1] == self.tokens_in_frame
            target_idx = target_idx.reshape(B, -1)[:, 1:-(target_idx.shape[-1] - 1)]

            # Remove extra frame from end we included for targets
            z_q_ind = z_q_ind[:,:-1,:]
            actions = actions[:,:-1]
            salience_level_ind = salience_level_ind[:,:-1]

            logits, expected_deviation = self.forward(
                z_q_ind, actions, salient_cluster_ind, salience_level_ind
            )


        # Calculate mean deviation loss for uncertainty ----------------------
        # Turn targets into one hot B x block_size x vocab_size with 1 in vocab

        one_hot = F.one_hot(target_idx, num_classes=self.output_size).squeeze()

        probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # Entropy across action-states
        wandb_log({'entropy/action-state': entropy.mean()})
        wandb_log({'probs_std': probs.std()})

        # Accuracy includes all sequence steps due to casual mask training
        acc = self.accuracy(
            logits.view(-1, logits.shape[-1]), target_idx.reshape(-1), split
        )
        if batch_idx % ACC_LOG_PERIOD == 0:
            log.info(f'Top x {split} prediction accuracy {acc}')
        p_diff = (one_hot - probs).abs()  # actual deviation
        d_diff = p_diff - expected_deviation
        d_loss = d_diff.square().sum() / d_diff.numel()

        # Calculate standard transformer categorical probability loss -----------
        # pytorch cross entropy has built-in softmax so pass logits
        p_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_idx.reshape(-1))
        loss = d_loss + p_loss
        self.iter += 1
        return {'loss': loss, 'logits': logits, 'expected_deviation': expected_deviation,
                'target_idx': target_idx}

    def s_i_to_as_i(self, state_idx, actions):
        """
        Expand state indexes into action-state indexes with dimension ordering S,A
        ----------------------------------------------
        Say we have a sequence of target state ints like

            0 1 2

        and we have 3 possible actions, we'd want to map the 3 states to 3 action-states for 9
        total action-states (a_s).

            s     a_s
            0 => 0 1 2
            1 => 3 4 5
            2 => 6 7 8
               ...

        so the possible action-states for a given state, s, are:
            a_s_all = [s * n + i for i in range(n)]
            e.g. a_s_all[0] = [0,1,2]

        Then the action-state index is defined as
            as_all + a
        ...where a is the action index

        Now say we took actions, 0 1 2, then the target idx would be 6,4,2:
            s,a       as   a    t
            2,0 => [6 7 8][0] = 6
            1,1 => [3 4 5][1] = 4
            0,2 => [0 1 2][2] = 2
        """
        target_idx = self.target_idx.to(state_idx.device)[state_idx * self.num_actions + actions]
        return target_idx

    def as_i_to_s_i(self, action_state_idx):
        """
        Action-state index to state index
        States are higher order dimension with all actions for each state stored sequentially
        """
        return action_state_idx // self.num_actions

    def as_i_to_a_i(self, action_state_idx):
        """
        Action-state index to action index
        States are higher order dimension with all actions for each state stored sequentially
        """
        return action_state_idx % self.num_actions

    def split_as_i(self, action_state_idx):
        """
        Like as_i_to_s_i, but returns the action and state (z_q) index
        """
        z_q_ind, action_i = divmod(action_state_idx, self.num_actions)
        return action_i, z_q_ind

    def accuracy(self, logits, target_idx, split):
        """logits (Batch, sequence Window, Action-State) """
        # TODO: State based accuracy (currently action-state)

        if not self.is_single_token2:
            top_acc_lvls = {
                '1': 1,
                '3': 3,
                'a': self.num_actions,
                '1pct': self.output_size // 100,
                '10pct': self.output_size // 10,
            }
            acc = self._get_acc(logits, split, target_idx, top_acc_lvls, 'acc')
            return sorted(list(zip(top_acc_lvls.values(), (float(x) for x in acc))))

        if self.is_single_token2:
            # Batch, State, Action dims
            B, S, A = logits.shape[0], logits.shape[1]//self.num_actions, self.num_actions
            assert self.output_size == A * S

            top_acc_lvls_as = {
                '1': 1,
                '3': 3,
                'a': A,
                '1pct': self.output_size // 100,
                '10pct': self.output_size // 10,
            }

            acc_as = self._get_acc(logits, split, target_idx, top_acc_lvls_as, 'acc_as')

            # Get state based accuracy, i.e. how accurate are we in predicting the state
            # that resulted from the chosen action
            action_taken = self.as_i_to_a_i(target_idx)
            target_states = self.as_i_to_s_i(target_idx)
            broadcast_target = torch.ones((B, S))
            target_actions = (broadcast_target.cuda() * action_taken.reshape(B, 1)).long()

            # Get logits of states for taken actions
            logits = logits.reshape(B, S, A)
            logits_states = logits.take_along_dim(target_actions.reshape(B, S, 1), dim=2).squeeze(-1)
            assert bool(logits[0][0][action_taken[0]] == logits_states[0][0]), 'Action selected for each batch'
            top_acc_lvls_s = {
                '1': 1,
                '3': 3,
                'a': A,  # Should be irrelevant as actions are factored out, but want to compare with action-state acc
                '1pct': S // 100,
                '10pct': S // 10,
            }
            acc_s = self._get_acc(logits_states, split, target_states, top_acc_lvls_s, 'acc_s')
            return sorted(list(zip(top_acc_lvls_s.values(), (float(x) for x in acc_s))))

    def _get_acc(self, logits, split, target_idx, top_acc_lvls, name):
        acc = accuracy(
            # rearrange combines batch and window dimensions into batch dimension
            # logits=rearrange(logits, 'd0 d1 d2 -> (d0 d1) d2'),
            # target=rearrange(target_idx, 'd0 d1 -> (d0 d1)'),
            logits=logits,
            target=target_idx,
            topk=top_acc_lvls.values(), )
        for lvl_i, lvl_name in enumerate(top_acc_lvls):
            wandb_log({f'{name}/{split}/top{lvl_name}': acc[lvl_i]})
        return acc

    def training_step(self, *args, **kwargs):
        return self.step_('train', *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.step_('val', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.step_('test', *args, **kwargs)

    def get_batch(self, batch):
        # TODO: Just put everything in agent_state, next_agent_state dicts
        if isinstance(batch, dict):
            batch_type = batch['type'][0]  # All same type
            return GPT_BATCH_TYPE_MAP[batch_type](**batch)
        elif len(batch) == 7:
            # TODO: We should be able to delete this now that we have GptBatch
            # Has agent state and indices
            (
                states,
                actions,
                rewards,
                dones,
                states_next,
                salience_level_ind,
                agent_states
            ) = batch
            z_q_ind = torch.stack([a['dvq_z_q_ind'] for a in agent_states])
            z_q_ind = z_q_ind.transpose(1, 0)
        else:
            raise NotImplementedError('Need to update this if we want to train on text again')
            # Text (i.e. not atari)
            actions = None  # This is for text so no action TODO: remove
            x, y = batch  # hate that i have to do this here in the model

        if self.is_single_token2:
            # Lightning collate does an annoying transpose on lists of dicts using zip() that switches B and S,
            # so permute from S,B => B,S
            z_q_flat = torch.stack([a['dvq_z_q_flat'] for a in agent_states])
            a_x, a_y, gpt_x, z_q_ind_x, z_q_ind_y = sa2as(z_q_flat, z_q_ind, actions)
            ret = [gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y, states, agent_states]
            return ret
        # else:
        #     z_q_ind.squeeze_(2)
        #     # z_q_flat.squeeze_(2)
        #     # z_q_flat = z_q_flat.permute(1, 0, 2, 3, 4)
        #
        #     # We don't really use z_q_flat in GPT anymore. We just need an extra cluster for the delim in z_q_ind.
        #     # TODO: (DONE?) We should get the delimiter and action embeddings from the same embedding as tokens, just shift
        #     #   the indexes. That way, we don't have as much overlap risk between the tensors, i.e. there's better
        #     #   separation between actions and states and delimiters. For training, we can mask out the non-action
        #     #   outputs when predicting actions, non-state when predicting states, and non-delim when predicting delim
        #     #   parts of a sequence.
        #     return z_q_ind, actions, salient_cluster_ind, salience_level_ind

    def add_non_state_tokens(
            self,
            z_q_ind,
            actions,
            salient_cluster_ind,
            salience_level_ind
    ):
        # Actions, delimiter, and salience
        return add_non_state_tokens(
            z_q_ind,
            actions,
            self.num_state_embeddings,
            self.num_actions,
            self.tokens_in_frame,
            salient_cluster_ind,
            salience_level_ind
        )
