import random
import numpy as np
import torch
from torch.nn import functional as F

from learn_max.constants import MAX_NUM_SALIENCE_LEVELS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)  # This is slow use random indexes instead  https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


def get_num_output_embeddings(num_state_embeddings: int, num_actions: int) -> int:
    """
    Number of embeddings for token types: state, action, and delim

    Adding more embeddings for salient tokens is done in the model
    """
    DELIM_TOKEN_TYPES = 1
    ret = num_state_embeddings + num_actions + DELIM_TOKEN_TYPES
    return ret


def get_action_and_delim_emb(actions, z_q_ind, z_q_emb, num_state_embeddings, num_actions, tokens_in_frame):
    # ret = tok_emb(z_q_ind)
    ret = quantizer.embed_code(z_q_ind)
    ret_emb_check = ret[:,:,:-2]
    assert ret[:,:,:-2] == z_q_emb.reshape(ret_emb_check.shape)
    return ret


def add_non_state_tokens(
        z_q_ind,
        actions,
        num_state_embeddings,
        num_actions,
        tokens_in_frame,
        salient_cluster_ind,
        salience_level_ind,
):
    device = z_q_ind.device
    if len(z_q_ind.shape) == 3:
        B, S, TiF = z_q_ind.shape  # batch, sequence-frames, tokens-in-frame
    elif len(z_q_ind.shape) == 2:
        B = 1
        S, TiF = z_q_ind.shape
    else:
        raise RuntimeError('Unexpected z_q_ind shape')
    # E = self.embedding_dim

    # Not using flat with patches as patches convey within-image info
    # flat_delim = self.tok_emb(torch.tensor(delim_ind).to(device))
    # flat_delim = flat_delim.repeat(B * S, 1)
    # z_q_flat = z_q_flat.reshape(B * S, H * W * E)
    # z_q_flat = torch.cat((z_q_flat, flat_delim), 1).reshape(B, S, H * W * E + E)

    salience_level_ind = salience_level_ind.reshape(B * S, 1)

    # Keep sensory and non-sensory tokens separate
    if salient_cluster_ind is None:
        action_z_q_ind = num_state_embeddings + actions
        action_z_q_ind = action_z_q_ind.reshape(B * S, 1)
        z_q_ind = z_q_ind.reshape(B * S, TiF)

        # After state patches and action token, delim token
        # is the same index for every frame
        delim_ind = num_state_embeddings + num_actions

        # index for delim to be fed into token embedding
        ind_delim = torch.tensor(delim_ind).to(device)
        ind_delim = ind_delim.repeat(B * S, 1)
        salience_level_ind += delim_ind  # avoid overlap with image patch indices
        gpt_ind = torch.cat(
            (z_q_ind, action_z_q_ind, salience_level_ind, ind_delim),
            -1,
        )
        gpt_ind = gpt_ind.reshape(B, S, tokens_in_frame)
    else:
        # TODO: Add index delim if poor salient prediction accuracy
        # We use a separate transformer for salient events, so don't need to worry about
        # overlap with image patch indices
        salient_cluster_ind += MAX_NUM_SALIENCE_LEVELS  # Don't overlap
        gpt_ind = torch.cat((salient_cluster_ind, salience_level_ind,), -1)



    assert gpt_ind.shape[-1] == tokens_in_frame

    return gpt_ind
