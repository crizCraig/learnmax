import torch

import numpy as np
from numpy import array

from queue import PriorityQueue


def learn_max_beam(seq, model, deviation, beam_batch_size=64):
    """
    Beam width should be batch size unless transformers combine batch + sequence trunk with the torch.tril mask

    We want total path interestingness (deviation) in the 50-70th percentile of all path costs.

    We should keep track of total interestingness along all paths searched, then go the 50-70th percentile of
    current paths.

    deviation = [
          0    1    2    3
    0    [0.1, 0.2, 0.3, 0.4],
    1    [0.2, 0.9, 0.1, 0.1],
    2    [0.1, 0.2, 0.1, 0.1],
    3    [0.2, 0.2, 0.3, 0.1],]

    #     0.6  1.5  0.8  0.7

    q = 0.2=(0,1), 0.3=(0,2)

    totals = [0.1, 1.1, 0.4, 0.4]

    q = 0.1=(2,2), 0.1=(1,3)

    totals = [0.1, 1.1, 0.5, 0.5]

    q = 0.1=(0,2), 0.1=(1,3)

    totals = [0.1, 1.1, 0.8, 0.6]

    probs for full input sequence (last in mask):

    seq=abcd

    i  action deviations
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

    Say these correspond to sequences, abcde and abcdf

    * take the transformer's input sequence of z(a,s) embeddings and output deviations as input
      (note that deviation is the predicted output prob change and is the proxy we are using for uncertainty)
    * get the action that corresponds to each output state by forwarding it through the dvq in batch(es) and getting
      the closest action embedding to the decoded one
    * sum the deviations over decoded actions to avoid searching over duplicate actions
    * sum these action_deviations (or possibly z deviation?) to the appropriate trajectories to get total deviation
      the network should be learning. we need to predict the state and the action in order to roll the model
      forward accurately, but deviation *should* be predicted to be the same across a specific action.
    * add these action_deviations to _all_ encountered sorted action_deviations with TORCH.SEARCHSORTED
        * insert new action_deviations with torch.cat - but also keep track of the associated sequence index in another
          (unsorted) dimension of the the action_deviations pool
        * sequences just get appended to and are not sorted, so sequence index can be static.
    * get topk_interesting(action_deviations, k) output indexes - (nb. interesting is defined as within 50-75pct
      uncertainty)
    * feed the corresponding z(a,s) embeddings at those indexes back into the transformer at the end of the
      current sequence (nb. we are simulating the env in the transformer)
    * get new states and deviations for the next level of the tree
    * when we build up a pool of transformer i/o that reaches some desired max size in GPU memory, get the most
      interesting trajectory and use the head z(a,s) of that trajectory to get our next action
    * use the z(a,s) => closest action that was obtained in summing action_deviations to decode the action
    * we can now throw away the search tree
    * once the action is taken, we will get one new state to train on
    * with that state, we shift the sequence window of the transformer forward and train one new batch
    * now get updated transformer output with the current sequence and can repeat the above
    * in tandem, we should be storing new observations from the environment in a pool that will be used as
      a batch to train the dvq online. this will mean that the embedding centroids will change and that the transformer
      will be associating moving centroids (different points) with the same output indexes over time. if this is
      a problem, we will need to train the dvq on bigger batches and try to pretrain it with some videos
      https://arxiv.org/abs/1805.11592 or pretrained agents perhaps
      (TODO: Ask Aditya Ramesh about this as he tried it with DALL-E and didn't notice improvment, but if it still
       worked then is great for us as our data is not I.I.D.).
    """

    # FIRST - we need to SUM the deviations by action - so the action for each vocab size in output needs to be computed
    #   IN A BATCH FROM THE DECODER!!!
    # action_deviation = sum_deviations_by_action()

    beam_i = topk_interesting(action_deviations, k=beam_batch_size)
    # get actions associated with beam_i using decoder IN A BATCH
    # add these actions to appropriate sequences
    # add the new deviations for actions _after_ i.




    #    0.6  0.8  0.8  0.7



    pass

def topk_interesting(action_deviations, k):
    """
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
    :param action_deviations: Predicted probability change
    :return: Indices of most interesting path heads
    """
    top = torch.topk(action_deviations, len(action_deviations) // 2 + 1, sorted=True)
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


if __name__ == '__main__':
    test_topk_interesting()
