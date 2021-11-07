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

if __name__ == '__main__':
    test_topk_interesting()
