import collections
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import List

import torch
import numpy as np
import wandb
from numpy import array
from torch import nn
from PIL import Image

from learn_max.constants import DATE_FMT, WANDB_MAX_LOG_PERIOD, ROOT_DIR, RUN_ID


def topk_interesting(entropy, k, rand_half=False):
    """
    Get top k actions with most entropy

    if rand_half: Pick random action in top 50-100 percentile entropy
    i     entropy
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

    Possible outputs are random samples without replacement of values between 0.6 and 1.0 so,

    0.6, 0.9
    or
    1.0, 0.6
    or
    0.8, 0.9

    :param k: Beam width
    :param entropy: Proxy for uncertainty / interesting-ness - shape: B,W,A
    :rand_half: If true, sample k randomly from top half of actions, else just get top k

    :return:
        actions: k action sequences (i.e. plans) - list of tuples of (batch, action index)
        action_entropy: k entropies of all dvq states across above actions
        actions_flat: k action indices indexing flattened array of length beam_size * num_actions
    """
    B, W, A = entropy.size()
    assert W == 1, 'Should only be looking at most recent state in window'
    entropy_flat = entropy.flatten()
    half_entropy = entropy_flat.numel() // 2
    assert half_entropy >= k, 'Not enough actions to sample k from top half'

    ent_coeff = -1 if 'REVERSE_ENTROPY' in os.environ else 1
    # Get highest indexes (could also use argsort)
    top = torch.topk(ent_coeff * entropy_flat, entropy_flat.numel()//2, sorted=True)

    if rand_half:
        # Pick k random actions from top half
        k_idx = torch.randperm(top.indices.size()[-1])[:k]
        actions_flat = top.indices[..., k_idx]
    else:
        actions_flat = top.indices[..., :k]  # Get first k as torch.topk sorts descending

    # Sort by index to recover batch order which allows vectorized head+tail concat of branches later on
    actions_flat, _ = actions_flat.sort()
    action_entropy = entropy_flat[actions_flat]  # top.values[..., k_idx]

    # Unflatten action indexes
    action_bi = torch.div(actions_flat, A, rounding_mode='floor')  # actions_flat // A => recover batch index
    action_ai = actions_flat - A * action_bi  # recover action index

    action_i = torch.stack((action_bi, action_ai)).T  # combine/zip up batch and action indexes

    # TODO: We perhaps want to re-weight entropy so that p75 is max interestingness as interestingness is not necessarily
    #   maximum entropy, but some tradeoff between novelty and predictability so that new information can be
    #   efficiently integrated with the current model (forming a natural curriculum),
    #   and with some option to anneal towards middle over time if the model capacity is reached in order
    #   to reduce forgetting. Also, this can be "fooled" into aleatoric traps like slot machines as they
    #   will always have high entropy. 
    assert len(action_i) == len(action_entropy) == len(actions_flat)

    return action_i, action_entropy, actions_flat


def test_topk_interesting():
    a, e, _ = topk_interesting(torch.arange(10).reshape(2, 1, 5), k=5)
    assert sorted(list(e.cpu().numpy())) == [5, 6, 7, 8, 9]
    assert all([b == 1 for b, a in a.cpu().numpy()])
    a, e, _ = topk_interesting(torch.arange(100).reshape(10, 1, 10), 10)
    assert not(set(list(e.cpu().numpy())) - set(range(50, 100)))
    assert all([b >= 5 for b, a in a.cpu().numpy()])


def get_action_states(logits, actions_flat):
    """
    Get most likely state as a result of taking a given action

    Params
    ------
    logits: B, W, A, |S|
    actions_flat: flat tensor of length beam_size * num_actions

    Returns
    -------
    Single dim tensor of length beam_batch_size - where we take the max probability state for each action in the last
    window of each batch

    Batches represent different paths taken throughout the planning tree, so the first time
    this is called, there's only one batch representing the trunk of the tree.
    """

    B, W, A, S = logits.size()
    assert W == 1
    logits = logits.reshape(B*A, S)

    logits_a = logits[actions_flat]
    s_idx = torch.argmax(logits_a, dim=1)
    return s_idx


def test_get_state():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    B, W, A, S, K = 4, 1, 18, 4096, 10  # batch, window, action, state, top_k actions
    logits = torch.rand(B, W, A, S)
    logits_all_batches = logits.reshape(B * A, S)
    actions_flat = torch.randint(0, B * A-1, (K,))
    a_s = get_action_states(logits, actions_flat)
    wi = 0

    def _test_action_state(bi, ai):
        """
        bi = batch index
        ai = action index
        """
        s_exp = torch.argmax(logits_all_batches[actions_flat][ai])
        s_actual = a_s[ai]
        assert s_exp == s_actual

    # Assert that logit for given action state is most likely
    _test_action_state(bi=0,  ai=0)
    _test_action_state(bi=-1, ai=-1)
    _test_action_state(bi=-1, ai=0)
    _test_action_state(bi=1, ai=1)


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


def _wandb_log_closure():
    """
    From: https://docs.wandb.ai/guides/track/log/logging-faqs
    We recommend that you try to log less than 10,000 points per metric. If you log more than 1 million points in a
    line, it will take us while to load the page. For more on strategies for reducing logging footprint without
    sacrificing accuracy, check out this Colab[1]. If you have more than 500 columns of config and summary metrics,
    we'll only show 500 in the table.

    [1] https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Logging_Strategies_for_High_Frequency_Data.ipynb#scrollTo=vDhYdZDR-7Er
    """
    accum = collections.defaultdict(list)

    # Start at 1 and increase to stay under 10k points per metric,
    # allows quickly seeing initial fast-changing metrics in wandb
    freq = 1

    def _wandb_log(msg_dict):
        nonlocal accum
        for k in msg_dict:
            accum[k].append(msg_dict[k])

    def check_flush(batch_idx):
        nonlocal accum, freq
        if batch_idx % freq != 0:
            return
        freq = min(int(freq + 0.1), WANDB_MAX_LOG_PERIOD)  # 10@1hz, 10@1/2hz, 10@1/3hz... 1/WANDB_MAX_LOG_PERIOD
        log_dict = {k: sum(accum[k]) / len(accum[k]) for k in accum}
        log_dict['batch_idx'] = batch_idx
        try:
            wandb.log(log_dict)
            accum.clear()
            return True
        except Exception as e:
            _wandb_log.error(f'Error logging to wandb {e}')
        return False
    return _wandb_log, check_flush


wandb_log, wandb_check_flush = _wandb_log_closure()


def sa2as(z_q_flat, z_q_ind, a):
    """
    Reorder gpt inputs/targets as action-states, i.e. action that leads to state, whereas inputs are state-actions,
    i.e. I'm in a state, take this action

    here we need to get the cluster indexes OR we could feed in the actual token as it already has semantic
    information and is the right size tensor. Regardless, the gpt targets will be integers.
    Feeding in the index allows the size of the token to vary.
    There's a question as to whether inputting centroids vs centroid indexes will make the model more
    robust to changes in centroids over time. It seems that the indexes are arbitrary, but they will
    be consistent most likely in terms of their semantic meaning. Although, feeding the whole centroid
    tensor would be even better.

    Okay, then we just need to shift the targets so that we are predicting the next token

    batch_size = batch[0].shape[0]

    80, 16, 1, 4410 => 16, 80, 4410
    Here we omit the first state with `1:` in order to pass action-states where the action leads to the state
    vs what we have now which are state-actions, where an action is taken in the state.
    We additionally index with `:-1` to keep the last state for the last target.
    in s:  s0 s1 s2 s3 s4
    in a:  a0 a1 a2 a3 a4

    out ax, sx, ay, sy
    -------------------
    ax: a0 a1 a2
    sx: s1 s2 s3

    ay: a1 a2 a3
    sy: s2 s3 s4
    """
    # TODO: Remove squeeze, doesn't work with batch size of 1 due to squeeze
    # TODO: BE SURE THIS ISN'T CAUSING AN EMBEDDING FROM A DIFFERENT RELATIVE TIMESTAMP VS FORWARD (i.e. state-action vs action-state)
    gpt_x = z_q_flat.squeeze().permute(1, 0, 2)[:, 1:-1, :]
    z_q_ind = z_q_ind.squeeze().permute(1, 0)[:, 1:]
    # z_q_ind = z_q_ind.view(batch_size, z_q_flat.shape[0] // batch_size, -1)
    z_q_ind_x = z_q_ind[:, :-1]
    z_q_ind_y = z_q_ind[:, 1:]  # GPT just predicts next state so we shift z_q_ind by one
    a_x = a[:, :-2]  # shift window left so we have action-states
    a_y = a[:, 1:-1]
    return a_x, a_y, gpt_x, z_q_ind_x, z_q_ind_y


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


def no_train(f):
    """
    Decorator for model ops that don't require grads and should be done in eval mode for things like Dropout
    """
    @wraps(f)
    def wrapper(self, *args, **kwds):
        with torch.no_grad():
            was_training = self.training
            self.eval()
            ret = f(self, *args, **kwds)
            if was_training:
                self.train()
        return ret
    return wrapper


# def get_viz_salience_folder():
#     f'{ROOT_DIR}/images/viz_salience/{get_date_str()}_r_{RUN_ID}_e_{self.current_epoch}'


def best_effort(f):
    """
    Decorator for model ops that don't require grads and should be done in eval mode for things like Dropout
    """
    @wraps(f)
    def wrapper(self, *args, **kwds):
        try:
            return f(self, *args, **kwds)
        except:
            print(traceback.format_exc())
    return wrapper


def dist(a, b):
    """
    Compute distance between all elements of a and b
    (a-b)^2

    a^2 - 2ab + b^2
    """
    return (
        a.pow(2).sum(1, keepdim=True)
        - 2 * a @ b.t()
        + b.pow(2).sum(1, keepdim=True).t()
    )


def torch_random_choice(tensor, salient_k):
    perm = torch.randperm(tensor.size(0))
    idx = perm[:salient_k]
    samples = tensor[idx]
    return samples, idx


def viz_experiences(experiences, folder, dvq_decoder, device, file_prefix=''):
    for i, x in enumerate(experiences):
        imo = viz_experience(x, dvq_decoder, device)
        filename = f'{folder}/{file_prefix + str(i).zfill(9)}.png'
        imo.save(filename)


def viz_experience(x, dvq_decoder, device):
    im = x.state.state.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # emb = x.state.dvq_z_q_flat.to(self.device)
    # B, H, W, E = emb.shape
    # if not self.is_single_token2:
    #     emb = emb.reshape(-1, self.dvq_embedding_dim)
    imz = dvq_decoder.forward(x.state.dvq_z_q_emb.to(device))

    # Image.fromarray(np.uint8(self.dvq.decoder.forward(x.state.dvq_z_q_emb.to(self.device)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255)).show()

    # imz = self.dvq.decode_flat(emb, output_proj=self.dvq.quantizer.output_proj)
    imz = imz.squeeze(0).permute(1, 2, 0).clamp(0, 1)


    # Combine patches into single image
    # imz = imz.reshape(H, W, *imz.shape[1:])
    # imz = imz.permute(0, 2, 1, 3, -1)
    # _, HP, _, WP, _ = imz.shape
    # imz = imz.reshape(H * HP, W * WP, -1)

    imz = imz.detach().cpu().numpy()
    imo = Image.fromarray(np.uint8(np.concatenate((im, imz), axis=0) * 255))
    return imo


def viz_salience(FiS, salient_replay_i, replay_buffers, dvq_decoder, device, file_prefix=''):
    os.makedirs(f'{ROOT_DIR}/images/viz_salience', exist_ok=True)
    folder_prefix = f'{ROOT_DIR}/images/viz_salience/{get_date_str()}_r_{RUN_ID}'
    for replay_i in salient_replay_i:
        folder = f'{folder_prefix}_i_{int(replay_i)}'
        os.makedirs(folder, exist_ok=True)
        print('Saving salience to ' + folder)
        # Use train_buf as we are sampling only from train w. train_only
        # TODO: Use viz_experiences and get all replay_i instead of this loop
        # Visualize a movie around the index
        viz_experiences(replay_buffers.train_buf.get(start=replay_i - FiS + 1, length=FiS * 2), folder,
                        dvq_decoder, device, file_prefix=file_prefix)


def dataclass_to_dict(thing, dataclass):
    if isinstance(thing, dataclass):
        return thing.dict()
    if isinstance(thing[0], dataclass):
        # agents_states only has a batch dimension, not window size
        return [_.dict() for _ in thing]

# TODO: Create GptBatchSensor and GptBatchSalient
@dataclass
class GptBatch:
    z_q_ind: torch.Tensor = None
    actions: torch.Tensor = None
    salient_event_ind: torch.Tensor = None
    salience_level_ind: torch.Tensor = None
    seq_len: int = None

    def __post_init__(self):
        self._ensure_shape()

    def dict(self):
        ret = dataclass_no_none_dict(self)
        ret['type'] = type(self).__name__

    def empty(self):
        self.z_q_ind = torch.tensor([0])
        self.actions = torch.tensor([0])
        self.salient_event_ind = torch.tensor([0])
        self.salience_level_ind = torch.tensor([0])
        self.seq_len = 0

    def __getitem__(self, batch_idx):
        if self.salient_event_ind is not None:
            salient_event_ind = self.salient_event_ind[batch_idx]
        else:
            salient_event_ind = None
        ret = GptBatch(
            z_q_ind=self.z_q_ind[batch_idx],
            actions=self.actions[batch_idx],
            salient_event_ind=salient_event_ind,
            salience_level_ind=self.salience_level_ind[batch_idx],
            seq_len=self.seq_len
        )
        ret.z_q_ind.squeeze_(0)
        if salient_event_ind is not None:
            ret.salient_event_ind.squeeze_(0)
        ret.actions.squeeze_(0)
        ret.salience_level_ind.squeeze_(0)
        return ret

    def append(self, other):
        if self.z_q_ind is None:
            assert self.actions is None
            assert self.salient_event_ind is None
            assert self.salience_level_ind is None
            self.z_q_ind = other.z_q_ind
            self.actions = other.actions
            self.salient_event_ind = other.salient_event_ind
            self.salience_level_ind = other.salience_level_ind
        else:
            assert None not in (
                self.actions, self.salient_event_ind, self.salience_level_ind
            )
            self.z_q_ind = torch.stack((self.z_q_ind, other.z_q_ind))
            self.actions = torch.stack((self.actions, other.actions))
            self.salient_event_ind = torch.stack(
                (self.salient_event_ind, other.salient_event_ind)
            )
            self.salience_level_ind = torch.stack((
                self.salience_level_ind,
                other.salience_level_ind),
                dim=1
            )

        self._ensure_shape()

    def _ensure_shape(self):
        if self.z_q_ind is not None:
            assert None not in (self.actions, self.salience_level_ind)
            a_shp = self.actions.shape
            if len(a_shp) == 1:
                if a_shp[0] != self.seq_len:
                    # We have multiple samples/sequences,
                    # add sequence dimension (and batch if needed)
                    self._reshape_multi_sequence(a_shp)
                else:
                    # Add batch dimension as 1 if missing
                    self._reshape_single_sequence()

    def _reshape_single_sequence(self):
        self.z_q_ind = self.z_q_ind.unsqueeze(0)
        self.actions = self.actions.unsqueeze(0)
        if self.salient_event_ind is not None:
            self.salient_event_ind = self.salient_event_ind.unsqueeze(0)
        self.salience_level_ind = self.salience_level_ind.unsqueeze(0)

    def _reshape_multi_sequence(self, a_shp):
        assert len(self.salience_level_ind.shape) == 1
        assert self.z_q_ind.shape[0] == a_shp[0]
        self.actions = self.actions.view(-1, self.seq_len)
        self.z_q_ind = self.z_q_ind.view(
            -1, self.seq_len, *self.z_q_ind.shape[-2:]
        )
        if self.salient_event_ind is not None:
            self.salient_event_ind = self.salient_event_ind.view(
                -1, self.seq_len
            )
        self.salience_level_ind = self.salience_level_ind.view(-1, self.seq_len)

    def num_steps(self):
        return len(self.salience_level_ind.view(-1))

    def __len__(self):
        # We always have salience_level_ind
        return len(self.salience_level_ind)


def dataclass_no_none_dict(obj):
    d = obj.__dict__
    ret = {}
    for k in d:
        if d[k] is not None:
            # Lightning doesn't want None's
            ret[k] = d[k]
    return ret

if __name__ == '__main__':
    test_get_state()
    test_topk_interesting()
