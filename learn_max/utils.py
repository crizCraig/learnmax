from datetime import datetime
from typing import List

import torch

import numpy as np
import wandb
from numpy import array
from torch import nn

from learn_max.constants import DATE_FMT


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


def get_date_str():
    return datetime.now().strftime(DATE_FMT)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    From: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b#gistcomment-3662283
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param logits: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        max_k = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top max_k indices that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top max_k scores just their corresponding indices/labels)
        _, y_pred = logits.topk(k=max_k, dim=1)  # _, [B, n_classes] -> [B, max_k]
        y_pred = y_pred.t()  # [B, max_k] -> [max_k, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in max_k values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target = target.contiguous()
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [max_k, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [max_k, B] where for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [max_k, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


if __name__ == '__main__':
    test_topk_interesting()
