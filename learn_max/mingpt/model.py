"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
from collections import defaultdict, deque
from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger as log
from torch.nn import functional as F

from learn_max.constants import ACC_LOG_PERIOD
from learn_max.utils import accuracy, wandb_log, sa2as, torch_random_choice


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
                 tokens_in_frame: int = None,  # number of tokens per frame (1 in single token)
                 batch_size: int = None,
                 ):
        super().__init__()
        self.output_size = output_size
        self.is_single_token2 = is_single_token2
        self.batch_size = batch_size
        self.frames_in_sequence_window = frames_in_sequence_window
        self.tokens_in_frame = tokens_in_frame

        # save these for optimizer init later
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        # input embedding stem: drop(content + position)
        #  Eventually:
        #     content = prev_salient + next_salient + dvq_token + learned_token + action_token
        action_embedding_dim = 100
        embedding_dim = input_embedding_dim  # + action_embedding_dim
        # self.dvq_proj = nn.Linear(input_embedding_dim, embedding_dim)
        assert embedding_dim % n_head == 0, f'Embedding len {embedding_dim} not evenly divisible by number of heads {n_head}'
        self.tok_emb = nn.Embedding(num_input_embeddings, input_embedding_dim)  # This should be num of z_q_emb,

        # block_size is the length of the model's context window in time
        if is_single_token2:
            seq_len = frames_in_sequence_window
        else:
            seq_len = tokens_in_frame * frames_in_sequence_window
            # Queue to store history of combined recent states for determining salience
            self.possibility_q = deque(maxlen=seq_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))
        self.patch_pos_emb = nn.Parameter(torch.zeros(1, tokens_in_frame, embedding_dim))
        self.frame_pos_emb = nn.Parameter(torch.zeros(1, frames_in_sequence_window, tokens_in_frame, embedding_dim))

        self.act_emb = nn.Embedding(num_actions, input_embedding_dim)
        # self.delim_emb = nn.Embedding(1, embedding_dim)

        self.drop = nn.Dropout(embd_pdrop)
        # deep transformer: just a sequence of transformer blocks
        self.blocks, _, out_dims = self.init_blocks(attn_pdrop, seq_len, embedding_dim, n_head, n_layer, resid_pdrop)
        out_embedding_dim = out_dims[-1]

        # decoder: at the end one more layernorm and decode the answers
        self.ln_f = nn.LayerNorm(out_embedding_dim)
        self.logit_p_head = nn.Linear(out_embedding_dim, output_size, bias=False)  # no need for extra bias due to one in ln_f

        # TODO: Remove or change to deviation over longer timescale - we want to know how the entropy of the
        #   probs changes over time so that, in the case of an aleotoric process like a slot machine, we see that
        #   while there may be patterns in recent data, the system over long stretches is random. The deviation head
        #   just predicts instantaneous changes to probability and so does not do this.
        self.deviation_head = nn.Linear(out_embedding_dim, output_size, bias=False)   # mean deviation

        self.target_idx = torch.arange(output_size)

        self.seq_len = seq_len
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
        self.salience_reservoir = deque(maxlen=10_000)
        self.salience_reservoir_n = 0

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
        return self.seq_len

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
            z_q_ind, replay_ind = args
            B, FiS, TiF = z_q_ind.size()  # batch, frames in sequence, tokens in frame
            assert TiF * FiS <= self.seq_len, "Cannot forward, model sequence length is exhausted."
            token_embed = self.tok_emb(z_q_ind)  # each input token index maps to a (learnable) vector
            assert FiS <= self.frames_in_sequence_window
            assert TiF <= self.tokens_in_frame

            # map frame and patch positions to a learnable vector
            frame_pos_embed = self.frame_pos_emb[:, :FiS]  # support partial sequences
            patch_pos_embed = self.patch_pos_emb[:, :TiF]  # support partial frames

            x = token_embed + frame_pos_embed + patch_pos_embed  # broadcast sum embeddings
            x = x.reshape(B, self.seq_len, self.embedding_dim)
            x = self.drop(x)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.logit_p_head(x)
            expected_deviation = self.deviation_head(
                x)  # Uses mean deviation instead of standard deviation https://stats.stackexchange.com/q/81986/18187
            wandb_log({'train/logits_std': logits.std()}, self.global_step)

            # Sum and normalize the tokens over sequence, i.e.
            #  emb_d * (num_state_positions (121) + 1 (action) + 1 (delim))
            #  in the window as the possibilities' regression target.

            # TODO: We need a buffer of previous possibilities
            #  of length block_size to detect saliency and a possibilities uncertainty target trained
            #  with possibilities loss to know how sure we are about saliency.

            possibilities = self.detect_salience(B, logits, replay_ind)

            return possibilities, logits, expected_deviation

    def detect_salience(self, B, logits, replay_ind):
        # Representation hierarchy is batch, sequences, frames, patches, logits
        # Sum the whole sequence of logits in order to get a description of what happened in the sequence
        # salience = ((salience - salience.min()) / max(1e-12, salience.max() - salience.min()))  # normalize 0=>1
        # This is a little better but still relative to min sum in batch
        # torch.log(logits.sum(axis=-1) - logits.sum(axis=-1).min() + 1e-12).min()
        S = self.seq_len
        FiS = self.frames_in_sequence_window
        TiF = self.tokens_in_frame
        L = logits.shape[-1]  # logit size
        # Deterministically squash saliency, i.e. not relative to current batch
        logits /= (S * L / 50)
        # TODO: As higher levels of abstraction are created, squash sums by dividing by some small epsilon plus
        #   the block size, i.e. salience = salience / (salience_lvl * block_size * epsilon)
        #   epsilon should be sized such that the max level of abstraction leads to numbers amenable to
        #   the optimizer. Should be easy to write some basic tests for this. This as the salience will be used
        #   as the embedding for the higher level. We could cluster the embedding and assign it an int
        #   value and remap it to a learned embedding to alleviate this. Then we just need to stay inside non-NaN
        #   range. We'd want to ensure the clustering is stable as new clusters are added, i.e. cluster 0 continues
        #   to represent the same types of things when adding new clusters.
        if logits.abs().max() > 5 or logits.abs().median() < 1e-5:
            log.warning(f'salience levels getting hard to optimize as embeddings '
                        f'max {logits.max()} '
                        f'median {logits.median()} '
                        f'mean {logits.mean()} '
                        f'min {logits.min()} '
                        )
        if B == 1 and not self.training:
            # Detect realtime salience with integration q
            self.possibility_q.append(logits)  # About 4.4MB with B=1,S=1107 * 4 bytes per int32 = B * S^2 * 4B
        elif self.training:
            # TODO: Detect across batch
            # Batch sequence needs to be sampled sequentially for this to work.
            # TODO:
            #   Slide the window across the batch and check for salience in train. Salience can pop up here
            #   when it didn't in realtime due to changing weights/logits

            assert S == logits.shape[-2], 'Could be partial window, but otherwise should ' \
                                                            'be true and windowing will be hard'
            # Sliding window across batch

            # key frames separated by sequence length to avoid overlap as nth frame logits
            # describe frame n-seq_len => n
            # windows = logits.flatten().unfold(dimension=0, size=self.tokens_in_frame, step=self.seq_len)
            windows = logits.flatten().unfold(dimension=0, size=S * L, step=TiF * L)
            assert (B-1) * S * L / (TiF * L) + 1 == windows.shape[0]  # sliding window check
            # Note we don't want to normalize this distance to the current batch as we want them to be comparable
            # across batches

            # manhattan distance between sequences shifted one sequence length apart
            salience = abs(windows[FiS:, :] - windows[:-FiS, :])

            # Sum salience across sequence
            salience = salience.sum(axis=1)

            assert int(replay_ind[1] - replay_ind[0]) == FiS
            # Interpolate replay indexes as we only have frame key indexes
            replay_ind = torch.arange(start=replay_ind[0], end=replay_ind[-1])
            replay_ind += self.frames_in_sequence_window - 1
            assert len(replay_ind) == len(salience) + FiS - 1, 'Last two sequences are used for last salience so we ' \
                                                              'have one fewer salient sequence than sequences in batch'
            # TODO: Look at top x% (abs?) and compare. We should look at the top 1% across more than just the batch.
            #   Ideally this is all time. So maybe reservoir sampling or just keep max N with some expiration.

            # Make sure reservoir is full
            if self.salience_reservoir.maxlen > len(self.salience_reservoir):
                num_to_insert = min(self.salience_reservoir.maxlen - len(self.salience_reservoir), len(salience))
                for i in range(num_to_insert):
                    self.salience_reservoir.append((replay_ind[i], salience[i]))
                    self.salience_reservoir_n += 1

            else:
                # Approx reservoir sampling probability for set with mean (believe this yields a recency bias)
                #   - prob of sampling one value from reservoir is k / n (see standard reservoir sampling)
                #   - the mean prob for set of samples with size m would then be:
                #     k * (1/n + 1/(n+1)... + 1/(n+m)) / m
                n = self.salience_reservoir_n
                m = salience.numel()
                k = int(0.1 * n)
                mean_prob = k * torch.sum(1 / torch.arange(n, n + m + 1)) / m
                salient_k = int(mean_prob * m)

                # We diverge from reservoir sampling here as we actually want the highest salience
                # experiences across a large number of batches. There will also be some duplicate
                # salient events detected which the distribution across batches helps a bit with
                # as we always sample some percentage of each batch. This as opposed to taking the most salient
                # across all batches which could result in many duplicates for the most salient events.
                samples, idx = torch.topk(salience, salient_k)
                for i, sample in enumerate(samples):
                    self.salience_reservoir.append((replay_ind[idx[i]], sample))

                self.salience_reservoir_n += m

                if n > 10 * self.salience_reservoir.maxlen:
                    # TOOD: Look at the most salient events
                    pass
            # The salience window index needs to be mapped back to a logits
            # batch/exp index with arg max that can be
            # visualized and eventually recognized as a recurring salient event

            # TODO: Try max cosine distance from (8) https://arxiv.org/pdf/2206.04114.pdf for detecting salient
            #   events

            # TODO: Randonmly sample mean_prob * batch length items from batch
            # TODO: High level context will be salient embeddings (these can be new clusters added to the
            #   existing trasformer softmax, a new transformer, or reused low level embeddings.
            #   These context tokens will have their own high level context slot (specifying the goal) for the
            #   low level. We may also have a prev context slot and num steps.
            #   Then for low level context, we detect when ANY salient event is encountered and update the input
            #   in the top level transformer when it is. This allows variable numbers of low level states to happen
            #   before a given salient (i.e. jumping up and down or taking a shorter or
            #   longer / circuitous route to a goal)
            #   For adding outputs to softmax, we can do net surgery or more simply have inactive outputs
            #   with logits forced to zero so no gradient flows until we allocate that output to something
            #   and stop zeroing its input logit.

            # TODO: Add salient events to replay buffers so that
            #  we can train with appropriate high level and low level context tokens. First
            #  let's just confirm we can detect salient events.
        return logits

    def single_token_forward(self, actions, embed, idx):
        # if self.should_input_embed:
        #     b, t, embed = idx_or_embed.size()
        # else:
        #     b, t = idx_or_embed.size()
        #     embed = None
        b, t, e = embed.size()
        assert t <= self.seq_len, "Cannot forward, model block size is exhausted."
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
        wandb_log({'train/logits_std': logits.std()}, self.global_step)
        return logits, expected_deviation

    def step_(self, split, batch, batch_idx=None):
        if self.is_single_token2:
            embed, z_q_ind, next_idx, a, a_next = batch  # gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y
            if split == 'train':
                assert embed.size()[1] == self.seq_len, \
                    'Not filling block size in train will result in untrained latter positions.'

            target_idx = self.s_i_to_as_i(next_idx, a_next)
            logits, expected_deviation = self.forward(embed, z_q_ind, a)  # B, S
        else:
            z_q_ind, exp_ind = batch[0]
            B, FiS, TiF = z_q_ind.size()  # batch, frames in sequence, tokens in frame
            if split == 'train':
                assert FiS * TiF >= self.seq_len, \
                    'Not filling block size in train will result in untrained latter positions.'
            # Shift targets by one patch
            target_idx = z_q_ind.reshape(B, -1)[:, 1:-(z_q_ind.shape[-1]-1)]

            z_q_ind = z_q_ind[:,:-1,:]  # Remove extra frame we included for targets
            salience, logits, expected_deviation = self.forward(z_q_ind, exp_ind)


        # Calculate mean deviation loss for uncertainty ----------------------
        # Turn targets into one hot B x block_size x vocab_size with 1 in vocab

        one_hot = F.one_hot(target_idx, num_classes=self.output_size).squeeze()

        probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # Entropy across action-states
        wandb_log({'entropy/action-state': entropy.mean()}, self.global_step)
        wandb_log({'probs_std': probs.std()}, self.global_step)

        # Use last position's logits as this is what we use for auto-regression / prediction (i.e. throwout
        #   most of the window generated with causal mask)
        #   Followup: Surprisingly makes little difference to accuracy
        last_logits = logits[:, -1, :]
        last_targets = target_idx[:, -1]
        acc = self.accuracy(last_logits, last_targets, split)
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
            wandb_log({f'{name}/{split}/top{lvl_name}': acc[lvl_i]}, self.global_step)
        return acc

    def training_step(self, *args, **kwargs):
        return self.step_('train', *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.step_('val', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.step_('test', *args, **kwargs)

    def get_batch_vars(self, batch):
        # TODO: Just put everything in agent_state, next_agent_state dicts
        if len(batch) == 7:
            # Has agent state and indices
            s, a, r, d, s_next, agent_state, indices = batch
        else:
            raise NotImplementedError('Need to update this if we want to train on text again')
            # Text (i.e. not atari)
            a = None  # This is for text so no action TODO: remove
            x, y = batch  # hate that i have to do this here in the model

        # TODO: Delete below once testing dvq training again
        z_q_ind = torch.stack([a['dvq_z_q_ind'] for a in agent_state])
        z_q_flat = torch.stack([a['dvq_z_q_flat'] for a in agent_state])

        if self.is_single_token2:
            a_x, a_y, gpt_x, z_q_ind_x, z_q_ind_y = sa2as(z_q_flat, z_q_ind, a)
            ret = [gpt_x, z_q_ind_x, z_q_ind_y, a_x, a_y, s, agent_state]
            return ret
        else:
            z_q_ind.squeeze_(2)
            z_q_flat.squeeze_(2)

            # Lightning collate does an annoying transpose on lists of dicts using zip() that switches B and S,
            # so permute from S,B => B,S
            z_q_ind = z_q_ind.permute(1, 0, 2, 3)
            z_q_flat = z_q_flat.permute(1, 0, 2, 3, 4)

            # Add frame delimiter
            # We don't really use z_q_flat in GPT anymore. We just need an extra cluster for the delim in z_q_ind.
            # TODO: We should get the delimiter and action embeddings from the same embedding as tokens, just shift
            #   the indexes. That way, we don't have as much overlap risk between the tensors, i.e. there's better
            #   separation between actions and states and delimiters. For training, we can mask out the non-action
            #   outputs when predicting actions, non-state when predicting states, and non-delim when predicting delim
            #   parts of a sequence.
            # TODO: Add within-frame position embeddings to dvq tokens for within image understanding
            # TODO: Add frame position embeddings
            device = z_q_ind.device
            B, S, H, W, E = z_q_flat.shape  # batch sequence-frames height width embedding
            delim_ind = self.num_state_embeddings + self.num_actions  # After state patches and action token

            # Not using flat with patches as patches convey within-image info
            # flat_delim = self.tok_emb(torch.tensor(delim_ind).to(device))
            # flat_delim = flat_delim.repeat(B * S, 1)
            # z_q_flat = z_q_flat.reshape(B * S, H * W * E)
            # z_q_flat = torch.cat((z_q_flat, flat_delim), 1).reshape(B, S, H * W * E + E)

            ind_delim = torch.tensor(delim_ind).to(device)  # add new cluster for delim
            ind_delim = ind_delim.repeat(B * S, 1)
            a_shifted = a + self.num_state_embeddings
            a_shifted = a_shifted.reshape(B * S, 1)
            z_q_ind = z_q_ind.reshape(B * S, H * W)
            z_q_ind = torch.cat((z_q_ind, a_shifted, ind_delim), -1)
            z_q_ind = z_q_ind.reshape(B, S, self.tokens_in_frame)
            return z_q_ind, indices
            # frame_len = ind.shape[-1]
            # ind = ind.reshape(B, -1)
            #
            #
            # # Create autoregression targets by shifting, we have an extra frame in case next patch is next frame,
            # #   but only need to predict one patch
            # gpt_ind_x = ind[:, :self.block_size]
            # gpt_ind_y = ind[:, 1:1+self.block_size]
            #
            # return gpt_ind_x, gpt_ind_y










        # s_ind =
        # z_q_flat shape single_token2 = 42, 2,  1, 4410 => S, B, 1, emb_d
        # z_q_flat shape patch based   = 42, 16, 1, 11, 11, 30 => S, B, 1, H, W, emb_d
        # The above H, W should be folded into S, but with some delimiter between frames




