import os
import random
import time

import torch
from tdigest import TDigest

from learn_max.mingpt.utils import add_action_and_delim


def detect_salience(actions, z_q_ind, replay_ind, seq_len, frames_in_sequence_window, tokens_in_frame,
                    num_state_embeddings, num_actions, tdigest, min_reservoir=1000):
    """
    Compare subsequent sequences of length frames_in_sequence_window. If the patch-wise difference
    is greater than the 90th percentile of previously seen data (approximated by a t-digest), then return the
    index at the end of the first sequence.
    """
    # Representation hierarchy is batch, sequences, frames, patches, logits
    # Sum the whole sequence of logits in order to get a description of what happened in the sequence
    # salience = ((salience - salience.min()) / max(1e-12, salience.max() - salience.min()))  # normalize 0=>1
    # This is a little better but still relative to min sum in batch
    # torch.log(logits.sum(axis=-1) - logits.sum(axis=-1).min() + 1e-12).min()
    S = seq_len
    FiS = frames_in_sequence_window
    TiF = tokens_in_frame

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
    # Note: Batch sequence needs to be sampled sequentially for this to work.
    # Slide the window across the batch and check for salience in train. Salience can pop up here
    #   when it didn't in realtime due to changing weights/logits
    B, _FiS, H, W = z_q_ind.shape
    assert B == 1  # and not self.training
    z_q_ind = z_q_ind.reshape(B, _FiS, H * W)
    z_q_ind = add_action_and_delim(actions, z_q_ind, num_state_embeddings, num_actions, tokens_in_frame)

    assert (B, _FiS, TiF) == z_q_ind.shape, 'No support for partial windows in salience detection'
    # Sliding window across batch
    # key frames separated by sequence length to avoid overlap as nth frame logits
    # use frames n-seq_len => n to output frame n+1
    # windows = logits.flatten().unfold(dimension=0, size=self.tokens_in_frame, step=self.seq_len)
    windows = z_q_ind.flatten().unfold(dimension=0, size=S, step=TiF)
    assert _FiS - FiS + 1 == windows.shape[0]  # sliding window check

    # Combine across sequence token-wise to see saliencies across the whole sequence.
    # This is basically saying that the spatial ordering of tokens is important for determining if we're
    # in a new situation, but not the temporal ordering. I.e., if the same frames occur, just in a different
    # temporal order, then the agent could just be walking around the same place in a different way.
    # So I don't want to count that. However, if the agent has moved some item, like a door or grabbed a key,
    # then new things have happened _spatially_ within the frame, and we DO want to count that.
    windows = windows.reshape(-1, FiS, TiF).transpose(2, 1)
    # We use geometric mean instead arithmetic mean and add 5e3 to get larger products.
    # These help reduce state aliasing when combining as there are fewer common factors than common summands.
    # Finally, we take the root first to avoid NaNs from prod.
    windows = ((windows + 5e3) ** (1 / FiS)).prod(dim=-1)

    # Note we don't want to normalize this distance to the current batch as we want them to be comparable
    # across batches

    # Manhattan distance between 2 sequences shifted to be one sequence length apart.
    # When only 2 sequence lengths are fed in, this just ends up taking the last minus first window.
    salience = abs(windows[FiS:, :] - windows[:-FiS, :])

    assert salience.shape[0] == _FiS - 2 * FiS + 1, 'Two sequences slid across input'

    # Sum diff across sequence-patches to get total salience for sequence
    salience = salience.sum(axis=-1)

    replay_ind = replay_ind.squeeze(0)
    # assert int(replay_ind[1] - replay_ind[0]) == FiS
    # # Interpolate replay indexes as we only have sequence start frame indexes
    # replay_ind = torch.arange(start=replay_ind[0], end=replay_ind[-1])
    # replay_ind += self.frames_in_sequence_window - 1
    # assert len(replay_ind) == len(salience) + FiS - 1, 'Last two sequences are used for last salience so we ' \
    #                                                    'have one fewer salient sequence than sequences in batch'
    # TODO: Look at top x% (abs?) and compare. We should look at the top 1% across more than just the batch.
    #   Ideally this is all time. So maybe reservoir sampling or just keep max N with some expiration.

    # TODO: Since salient events are based on dvq patch diffs over sequences, they are agnostic to GPT.
    #   Therefore we only need to look at a given index ONCE! This can be when it is observed!

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
    #   However, by combining many examples of the same salient state, we should be able to get something similar.
    #   The problem is that the state might not be captured as salient unless all possibilties are considered.

    # TODO: We should include semantic data (logits/patch embeddings) in salient state embeddings to allow for task transfer
    #   I.e. imagine the skull is blue and slightly larger, or there's a rolling ball. The knowledge on how to
    #   jump over the skull should transfer to the new object.

    salience = salience.detach().cpu().numpy()
    ret = []
    if tdigest.n > min_reservoir:  # Get a good population before sampling top percentile
        for i, s in enumerate(salience):
            if 'LEAST_SALIENT' in os.environ:
                is_salient = s < tdigest.percentile(10)
            else:
                is_salient = s > tdigest.percentile(90)
            if is_salient:
                # TODO: Look at the most salient events
                # TODO: We should append more than one frame for the event so that we can more
                #   easily detect duplicate salient events. This as the middle frame may be different
                #   due to slightly different states/actions before and after, but the essential event is
                #   the same. Also, it will be crucial to find the shortest path to the event in order to
                #   train lower level actions with the event as the goal context. RL _could_ be used to
                #   find the shortest path as well once the event is known, esp with a bellman-backup method
                #   like q-learning.
                ret.append(replay_ind[i] + FiS - 1)

    if len(salience) == 1:
        tdigest.update(salience[0])
        if random.random() < 0.01:
            tdigest.compress()
    elif len(salience) > 1:
        tdigest.batch_update(salience)  # always does compress
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

    # TODO: Visualize salient events

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


def _test_detect_salience(num_sequences=2):
    frames_in_sequence_window = 8
    num_actions = 6
    num_frames = frames_in_sequence_window * num_sequences
    actions = torch.randint(low=0, high=num_actions-1, size=(1, num_frames))
    num_state_embeddings = 256
    patches_per_side_of_frame = 11
    tokens_in_frame = 123
    z_q_ind = torch.randint(
        low=0,
        high=num_state_embeddings,
        size=(1,
              num_sequences * frames_in_sequence_window,
              patches_per_side_of_frame,
              patches_per_side_of_frame))
    replay_ind = torch.arange(0, num_frames).unsqueeze(0)
    seq_len = frames_in_sequence_window * tokens_in_frame
    tdigest = TDigest()
    min_reservoir = 10
    for i in range(min_reservoir * 2):
        # Fill up with zeroes
        detect_salience(actions, 0 * z_q_ind, replay_ind, seq_len, frames_in_sequence_window, tokens_in_frame,
                        num_state_embeddings, num_actions, tdigest, min_reservoir=min_reservoir)
    salient_i = detect_salience(actions, z_q_ind, replay_ind, seq_len, frames_in_sequence_window, tokens_in_frame,
                                num_state_embeddings, num_actions, tdigest, min_reservoir=min_reservoir)
    assert len(salient_i) == num_frames - 2 * frames_in_sequence_window + 1, 'Two sequences slid across input'
    return salient_i


def test_2_sequences():
    salient_replay_i = _test_detect_salience(num_sequences=2)
    assert salient_replay_i == [7]


def test_3_sequences():
    salient_replay_i = _test_detect_salience(num_sequences=3)
    assert salient_replay_i == [7,8,9,10,11,12,13,14,15]


def test_all():
    if 'LEAST_SALIENT' in os.environ:
        return
    start = time.time()
    test_2_sequences()
    test_3_sequences()
    print(f'Tested salience detection in {round(1e3*(time.time() - start))}ms')


# Run on import so tests stay up to date
test_all()
