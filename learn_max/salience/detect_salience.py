import os
import random
import time

import numpy as np
import torch
from tdigest import TDigest
from torch import nn

from loguru import logger as log

from learn_max.constants import NUM_DIFF_SEQ
from learn_max.dvq.model.quantize import VQVAEQuantize
from learn_max.mingpt.utils import get_num_output_embeddings
from learn_max.utils import wandb_log


@torch.no_grad()
def detect_salience(
    actions,
    z_q_ind,
    z_q_emb,
    replay_ind,
    seq_len,
    frames_in_sequence_window,
    state_tokens_in_frame,
    tokens_in_frame,
    num_state_embeddings,
    num_actions,
    tdigest,
    min_reservoir=1000,
    use_emb=True,
    logits=None,
):
    """
    Compare subsequent sequences of length frames_in_sequence_window. If the patch-wise difference
    is greater than the 90th percentile of previously seen data (approximated by a t-digest), then return the
    index at the end of the first sequence.

    Expect either logits or use_emb, not both or neither (xor)

    @param logits: Allows detecting a big change in the predicted future not just the current observed state
    """
    # Representation hierarchy is batch, sequences, frames, patches, logits
    # Sum the whole sequence of logits in order to get a description of what happened in the sequence
    # salience = ((salience - salience.min()) / max(1e-12, salience.max() - salience.min()))  # normalize 0=>1
    # This is a little better but still relative to min sum in batch
    # torch.log(logits.sum(axis=-1) - logits.sum(axis=-1).min() + 1e-12).min()
    # logits: 14, 8*123=984, 263
    FiS = frames_in_sequence_window
    # actions part of state with logits (FINE)
    TiF = state_tokens_in_frame if logits is None else tokens_in_frame
    S = FiS * TiF

    if logits is None and use_emb is False:
        raise RuntimeError('Int (encoder cluster centroid) mode not supported')

    # Deterministically squash saliency, i.e. not relative to current batch
    # z_q_ind /= (S * Z / 50)
    # TODO: As higher levels of abstraction are created, squash sums by dividing by some small epsilon plus
    #   the block size, i.e. salience = salience / (salience_lvl * block_size * epsilon)
    #   epsilon should be sized such that the max level of abstraction leads to numbers amenable to
    #   the optimizer. Should be easy to write some basic tests for this. This as the salience will be used
    #   as the embedding for the higher level. We could cluster the embedding and assign it an int
    #   value and remap it to a learned embedding to alleviate this. Then we just need to stay inside non-NaN
    #   range. We'd want to ensure the clustering is stable as new clusters are added, i.e. cluster 0 continues
    #   to represent the same types of things when adding new clusters.
    # if z_q_ind.abs().max() > 5 or z_q_ind.abs().median() < 1e-5:
    #     log.warning(f'salience levels getting hard to optimize as embeddings '
    #                 f'max {z_q_ind.max()} '
    #                 f'median {z_q_ind.median()} '
    #                 f'mean {z_q_ind.mean()} '
    #                 f'min {z_q_ind.min()} '
    #                 )
    # Note: Input sequences needs to be sampled sequentially for this to work.
    # Slide the window across the batch and check for salience in train. Salience can pop up here
    #   when it didn't in realtime due to changing weights/logits
    B, _FiS, H, W = z_q_ind.shape
    if use_emb:
        E = z_q_emb.shape[-1]
    L = None
    z_q_ind = z_q_ind.reshape(B, _FiS, H * W)

    # We don't want to add the action and delim tokens here because we want salience recognition to be agnostic to
    # actions and delims are implicit in the way diffs are done.
    # z_q_ind = add_action_and_delim_ind(actions, z_q_ind, num_state_embeddings, num_actions, tokens_in_frame)
    if logits is not None:
        assert not use_emb, 'Expect either logits or use_emb, but both set'
        L = logits.shape[-1]
        x = logits.reshape(B, _FiS, -1, L)
    elif use_emb:
        # Set x to embedding so that distances are meaningful in input space
        x = z_q_emb.reshape(B, _FiS, H * W, E)
    else:
        x = z_q_ind
    assert (B, _FiS, TiF) == x.shape[:3], 'No support for partial windows in salience detection'

    assert B == 1

    if logits is not None:
        x = x.reshape(B * _FiS * TiF, L)
    elif use_emb:
        x = x.reshape(B * _FiS * TiF, E)
    else:
        x = x.flatten()

    # Slide sequence window across tokens/patches one frame at a time
    windows = x.unfold(dimension=0, size=S, step=TiF)
    assert _FiS - FiS + 1 == windows.shape[0]  # sliding window check

    # Combine across sequence token-wise to see saliencies across the whole sequence.
    # ------------------------------------------------------------------------------------------------------------------
    # This is basically saying that the spatial ordering of tokens is important for determining if we're
    # in a new situation, but not the temporal ordering. I.e., if the same frames occur, just in a different
    # temporal order, then the agent could just be walking around the same place in a different way.
    # So I don't want to count that. However, if the agent has moved some item, like a door or grabbed a key,
    # then new things have happened _spatially_ within the frame, and we DO want to count that.
    # TODO: Remove these redundant if blocks
    if logits is not None:
        windows = windows.reshape(windows.shape[0], L, FiS, TiF)
    elif use_emb:
        windows = windows.reshape(windows.shape[0], E, FiS, TiF)
    else:
        windows = windows.reshape(windows.shape[0], FiS, TiF)
    # windows = windows.transpose(1, 3)
    # We add 5e3 and use geometric mean instead of arithmetic mean to get more separation.
    # These help reduce state aliasing when combining as there are fewer common factors than common summands.
    # Finally, we take the root first to avoid NaNs from prod.
    if use_emb or logits is not None:
        combine_dim = -2
    else:
        combine_dim = -1
    windows = ((windows + 5e3) ** (1 / FiS)).prod(dim=combine_dim)
    # ------------------------------------------------------------------------------------------------------------------

    # Note we don't want to normalize this distance to the current batch as we want them to be comparable
    # across batches

    # Manhattan distances between subsequent sequences
    #   E.g. say our sequence length is 8 frames and our input has 16 (_FiS) total frames, so 2 sequences.
    #   Then we just subtract the first sequence from the second.
    #   |________;________|  => B - A
    #       A         B
    #   However, you'd have 9 windows since you start at the first 8 frames (1 window) and then step the window
    #   8 more times (+ 8 windows) until the end of the window reaches the 16th frame when you use `unfold`
    #   Most of these windows are not used when only 2 sequences are fed in,
    #   but if even one more frame is fed in, we'll start utilizing a greater percentage of the windows.
    #   E.g. say you feed in 17 frames, then you'd get 10 windows and 2 salience diffs (20% usage, from 11%),
    #   and if you fed in 24 frames, you'd get 17 windows and 8 salience diffs (47% usage).
    salience = abs(windows[FiS:] - windows[:-FiS])

    assert salience.shape[0] == _FiS - NUM_DIFF_SEQ * FiS + 1, 'Two subsequent sequences slid across windows'

    # Sum diff across sequence-patches to get total salience for sequence
    salience = salience.reshape(salience.shape[0], -1).sum(axis=-1)

    replay_ind = replay_ind.squeeze(0)
    # assert int(replay_ind[1] - replay_ind[0]) == FiS
    # # Interpolate replay indexes as we only have sequence start frame indexes
    # replay_ind = torch.arange(start=replay_ind[0], end=replay_ind[-1])
    # replay_ind += self.frames_in_sequence_window - 1
    # assert len(replay_ind) == len(salience) + FiS - 1, 'Last two sequences are used for last salience so we ' \
    #                                                    'have one fewer salient sequence than sequences in batch'

    # TODO: We need to plan not just to get a large distance away from the centroid of previously encountered
    #   events at the highest level of saliency, BUT to also explore and resolve the most uncertainty possible
    #   at the highest level of saliency.

    # TODO: Should we integrate logits back in to give a better sense of what's reachable from a given state per
    #   the model? This allows saying what's _possible_ now that I opened the door for example. To do this
    #   we should combine a next frame prediction (all patches) autoregressively created, with a nth (say 10th) frame
    #   prediction. The probs we care about are patch-embeddings, so 123 * 263 probs, which describe the
    #   possibilities reachable from the state. Then we can detect when the possibilities change drastically,
    #   frame to frame. Since this depends on the network, we should also go back and detect salience on older
    #   data in case the network has learned new possibilities. Alternatively, we could wait to encounter the event again.
    #   ----
    #   This allows aggregating what's possible after a state from experience, not just what happened once.
    #   However, by combining many examples of the same salient state without logits,
    #   we should be able to get something similar. The problem is that the state might not be captured as salient
    #   unless all possibilties are considered.

    # TODO: We should include semantic data (logits/patch embeddings) in salient state embeddings to allow for task transfer
    #   I.e. imagine the skull is blue and slightly larger, or there's a rolling ball. The knowledge on how to
    #   jump over the skull should transfer to the new object.

    salience = salience.detach().cpu().numpy()
    ret = []
    if tdigest.n < min_reservoir and tdigest.n % 100 == 0:
        # Log reservoir percentage
        log.info(f'Building reservoir: {tdigest.n / min_reservoir * 100:.2f}%')
    if tdigest.n > min_reservoir:  # Get a big pool before sampling top percentile
        for i, s in enumerate(salience):
            # TODO: Test tdigest 10pct, 90pct with random numbers, should not increase!!!
            tdigest_50pct = tdigest.percentile(50)
            tdigest_90pct = tdigest.percentile(90)
            wandb_log({'salience/salience_candidate': s})
            wandb_log({'salience/tdigest_min': tdigest.percentile(0)})
            wandb_log({'salience/tdigest_50pct': tdigest_50pct})
            wandb_log({'salience/tdigest_90pct': tdigest_90pct})
            if 'LEAST_SALIENT' in os.environ:
                is_salient = s < tdigest_50pct
            else:
                is_salient = s > tdigest_90pct
            if is_salient:
                before_salient = windows[i]
                after_salient = windows[i + FiS]
                patch_diff = after_salient - before_salient

                # check that recomputing patch diff for top saliences is correct
                assert patch_diff.abs().sum() == s
                ret.append(
                    # mid_replay_ind is the last index of the first sequence
                    # TODO: Make this a dataclass instead of a dict
                    dict(mid_replay_ind=replay_ind[i] + FiS - 1, patch_diff=patch_diff)
                )

    if len(salience) > 0:
        if len(salience) == 1:
            tdigest.update(salience[0])
            if random.random() < 0.01:
                tdigest.compress()  # Perform tdigest maintenance once in a while
        else:
            tdigest.batch_update(salience)  # this always does maintenance
    # else:
    #     # Approx reservoir sampling probability for set with mean (believe this yields a slight recency bias vs
    #     # sampling one at a time)
    #     #   - prob of sampling one value from reservoir is k / n (see standard reservoir sampling)
    #     #   - the mean prob for set of samples with size m would then be:
    #     #     k * (1/n + 1/(n+1)... + 1/(n+m)) / m
    #     n = self.salience_reservoir_n
    #     m = salience.numel()
    #     k = int(0.1 * n)
    #     mean_prob = k * torch.sum(1 / torch.arange(n, n + m + 1)) / m
    #     salient_k = int(mean_prob * m)
    #     if salient_k < 1:
    #         log.warning('salient_k less than 1, ')
    #
    #     # We diverge from reservoir sampling here as we actually want the most salient
    #     # experiences across a large number of batches.
    #     samples, idx = torch.topk(salience, salient_k)
    #     for i, sample in enumerate(samples):

    #         self.tdigest.append((replay_ind[idx[i]], sample))
    #
    #     self.salience_reservoir_n += m
    #
    #     if n > 10 * self.tdigest.maxlen:
    #         most_salient = torch.topk(torch.tensor(self.tdigest)[:, 1], 1000, dim=0)

    # TODO: Maximize a total salient distance in the planning horizon to avoid loops
    #  Uncertainty is better than total distance though, as it also allows back tracking.

    # TODO: Try max cosine distance from (8) https://arxiv.org/pdf/2206.04114.pdf for detecting salient
    #   events

    # TODO: High level context will be salient embeddings (these can be new clusters added to the
    #   existing trasformer softmax, a new transformer, or reused low level embeddings.
    #   These context tokens will have their own high level context slot (specifying the goal) for the
    #   lower level. We may also have a prev context slot and num steps.
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
    return ret


def _test_detect_salience(num_sequences=2, use_emb=False):
    frames_in_sequence_window = 8
    num_actions = 6
    num_frames = frames_in_sequence_window * num_sequences
    actions = torch.randint(low=0, high=num_actions-1, size=(1, num_frames))
    num_state_embeddings = 256
    patch_width = 11
    state_tokens_in_frame = 121
    tokens_in_frame = state_tokens_in_frame + 2  #  + 1 action + 1 delimiter
    embedding_size = 30
    batch_size = 1
    # num_input_embeddings = get_num_embeddings(num_state_embeddings, num_actions)

    z_q_ind = torch.randint(
        low=0,
        high=num_state_embeddings,
        size=(
            batch_size,
            num_sequences * frames_in_sequence_window,
            patch_width,
            patch_width,
        ),
    )
    if use_emb:
        logits = None
        z_q_emb = -1 + 2 * torch.rand(
            batch_size,
            num_sequences * frames_in_sequence_window,
            patch_width,
            patch_width,
            embedding_size,
        )
    else:
        z_q_emb = None
        logits = -1 + 2 * torch.rand(
            batch_size,
            num_sequences * frames_in_sequence_window,  # 2 * 8
            tokens_in_frame,
            get_num_output_embeddings(num_state_embeddings, num_actions),
        )
    # Logit shape: B, _FiS, state_tokens_in_frame, L (263???)  1,16,123,263

    replay_ind = torch.arange(0, num_frames).unsqueeze(0)
    seq_len = frames_in_sequence_window * state_tokens_in_frame
    tdigest = TDigest()
    min_reservoir = 10
    for i in range(min_reservoir * 2):
        # Fill up with zeroes
        zero_z_q_emb = torch.zeros_like(z_q_emb) if use_emb else None
        zero_logits = torch.zeros_like(logits) if not use_emb else None
        detect_salience(
            actions,
            0 * z_q_ind,
            zero_z_q_emb,
            replay_ind,
            seq_len,
            frames_in_sequence_window,
            state_tokens_in_frame,
            tokens_in_frame,
            num_state_embeddings,
            num_actions,
            tdigest,
            min_reservoir=min_reservoir,
            use_emb=use_emb,
            logits=zero_logits,
        )
    ret = detect_salience(
        actions,
        z_q_ind,
        z_q_emb,
        replay_ind,
        seq_len,
        frames_in_sequence_window,
        state_tokens_in_frame,
        tokens_in_frame,
        num_state_embeddings,
        num_actions,
        tdigest,
        min_reservoir=min_reservoir,
        use_emb=use_emb,
        logits=logits,
    )
    assert (
        len(ret) == num_frames - 2 * frames_in_sequence_window + 1
    ), 'Two sequences slid across input'
    return ret


def test_2_sequences():
    salience = _test_detect_salience(num_sequences=2)
    assert salience[0]['mid_replay_ind'] == 7


def test_3_sequences():
    salience = _test_detect_salience(num_sequences=3)
    assert [s['mid_replay_ind'] for s in salience] == [7,8,9,10,11,12,13,14,15]


def test_2_sequences_emb():
    salience = _test_detect_salience(num_sequences=2, use_emb=True)
    assert salience[0]['mid_replay_ind'] == 7


def test_3_sequences_emb():
    salience = _test_detect_salience(num_sequences=3, use_emb=True)
    assert [s['mid_replay_ind'] for s in salience] == [7,8,9,10,11,12,13,14,15]


def test_all():
    if 'LEAST_SALIENT' in os.environ or 'DISABLE_SALIENCE_TESTS' in os.environ:
        return
    start = time.time()
    test_3_sequences_emb()
    test_2_sequences_emb()
    test_3_sequences()
    test_2_sequences()
    log.success(f'Tested salience detection in {round(1e3*(time.time() - start))}ms')


def test_emb():
    test_3_sequences_emb()
    test_2_sequences_emb()


if 'TEST_SALIENCE_EMB' in os.environ:
    test_emb()
else:
    # Run on import so tests stay up to date
    test_all()
