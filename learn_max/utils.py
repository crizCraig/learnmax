import torch

import numpy as np
import wandb
from numpy import array
from torch import nn


def topk_interesting(deviations, k):
    """
    i    deviations
    0:      0.1
    1:      0.2
    2:      0.3
    3:      0.4
    4:      0.5
    5:      0.6
    6:      0.7
    7:      0.8
    8:      0.9
    9:      1.0

    If k == 2, then we take index 5 and 6

    Algorithm:
    k = 2
    deviation = torch.arange(10)
    >> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    top = torch.topk(deviation, len(deviation)//2 + 1, sorted=True)
    >> torch.return_types.topk(
    >> values=tensor([9, 8, 7, 6, 5, 4]),
    >> indices=tensor([9, 8, 7, 6, 5, 4]))

    mid = torch.topk(top.indices, len(top.indices)//2, sorted=True, largest=False)
    >> torch.return_types.topk(
    >> values=tensor([4, 5, 6, 7]),
    >> indices=tensor([5, 4, 3, 2]))

    # Now values are indices and we want to return the middle k indices, 5 and 6
    chop = len(mid.values) - k
    return mid.values[chop//2:-chop//2]  => (5,6)

    :param k: Beam width
    :param deviations: Predicted probability change
    :return: Indices of most interesting path heads
    """
    top = torch.topk(deviations, len(deviations) // 2 + 1, sorted=True)
    mid = torch.topk(top.indices, int(len(top.indices)/2) + 1, sorted=True, largest=False)

    # mid's values are top's indices
    chop = len(mid.values) - k
    return mid.values[chop//2:-chop//2]


def test_topk_interesting():
    r = topk_interesting(torch.arange(10), 2)
    assert list(r) == [5, 6]
    r = topk_interesting(torch.arange(100), 10)
    assert list(r) == [57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
    r = topk_interesting(torch.arange(512), 10)
    assert list(r) == [314, 315, 316, 317, 318, 319, 320, 321, 322, 323]


# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - row[j]]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


def main_example():
    # define a sequence of 10 words over a vocab of 5 words
    data = -np.log([[0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1],
                   [0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1],
                   [0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1],
                   [0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1],
                   [0.1, 0.2, 0.3, 0.4, 0.5],
                   [0.5, 0.4, 0.3, 0.2, 0.1]])
    data = array(data)
    # decode sequence
    result = beam_search_decoder(data, 3)
    # print result
    for seq in result:
        print(seq)

def _init_weights(module):
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


def wandb_try_log(msg_dict):
    try:
        wandb.log(msg_dict)
    except:
        pass


def get_batch_vars(batch, use_next=False, return_agent_state=False, populate_gpt=False):
    agent_state = None
    if len(batch) == 5:
        # No agent state
        s, a, r, d, s_next = batch
        if use_next:
            dvq_x = torch.cat((s, s_next))  # doubles batch size which we don't want with GPT as there's OOM
        else:
            dvq_x = s
    elif len(batch) == 6:
        # Has agent state
        s, a, r, d, s_next, agent_state = batch
        dvq_x = s
    else:
        # Text (i.e. not atari)
        a = None  # This is for text so no action TODO: remove
        dvq_x, y = batch  # hate that i have to do this here in the model
    if not populate_gpt:
        dvq_batch = dvq_x
        if return_agent_state:
            dvq_batch = dvq_x, agent_state
        return dvq_batch
    else:
        dvq_loss = torch.mean(torch.Tensor([a['dvq_loss'].mean() for a in agent_state]))
        # dvq_x_hat = torch.Tensor([a['dvq_x_hat'] for a in agent_state])
        z_q_ind = torch.stack([a['dvq_z_q_ind'] for a in agent_state])
        z_q_flat = torch.stack([a['dvq_z_q_flat'] for a in agent_state])

        # print(f'{dvq_loss=}')
        # print(f'dvq_loss_avg={sum([a["dvq_loss"].mean() for a in agent_state]) / len(agent_state)}')

        # here we need to get the cluster indexes OR we could feed in the actual token as it already has semantic
        # information and is the right size tensor. Regardless, the gpt targets will be integers.
        # Feeding in the index allows the size of the token to vary.
        # There's a question as to whether inputting centroids vs centroid indexes will make the model more
        # robust to changes in centroids over time. It seems that the indexes are arbitrary, but they will
        # be consistent most likely in terms of their semantic meaning. Although, feeding the whole centroid
        # tensor would be even better.

        # Okay, then we just need to shift the targets so that we are predicting the next token

        batch_size = batch[0].shape[0]

        # 80, 16, 1, 4410 => 16, 80, 4410
        gpt_x = z_q_flat.squeeze().permute(1, 0, 2)[:, :-1, :]
        # gpt_x = z_q_flat.view(batch_size, z_q_flat.shape[0] // batch_size, -1)[:, :-1, :]

        z_q_ind = z_q_ind.squeeze().permute(1, 0)
        # z_q_ind = z_q_ind.view(batch_size, z_q_flat.shape[0] // batch_size, -1)
        z_q_ind_x = z_q_ind[:, :-1]
        z_q_ind_y = z_q_ind[:, 1:]
        return gpt_x, z_q_ind_x, z_q_ind_y, a[:, :-1], s
        # idx_or_embed = idx_or_embed.view(int(idx_or_embed.shape[0] / self.block_size) - 1, self.block_size,
        #                                  idx_or_embed.shape[1])


if __name__ == '__main__':
    test_topk_interesting()
