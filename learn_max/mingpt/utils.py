import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


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


def get_num_embeddings(num_state_embeddings, num_actions):
    return num_state_embeddings + num_actions + 1  # actions + 1 for frame delimiter


def get_action_and_delim_emb(actions, z_q_ind, z_q_emb, num_state_embeddings, num_actions, tokens_in_frame):
    # ret = tok_emb(z_q_ind)
    ret = quantizer.embed_code(z_q_ind)
    ret_emb_check = ret[:,:,:-2]
    assert ret[:,:,:-2] == z_q_emb.reshape(ret_emb_check.shape)
    return ret


def add_action_and_delim_ind(actions, z_q_ind, num_state_embeddings, num_actions, tokens_in_frame):
    device = z_q_ind.device
    B, S, TiF = z_q_ind.shape  # batch sequence-frames height width embedding
    # E = self.embedding_dim
    delim_ind = num_state_embeddings + num_actions  # After state patches and action token
    # Not using flat with patches as patches convey within-image info
    # flat_delim = self.tok_emb(torch.tensor(delim_ind).to(device))
    # flat_delim = flat_delim.repeat(B * S, 1)
    # z_q_flat = z_q_flat.reshape(B * S, H * W * E)
    # z_q_flat = torch.cat((z_q_flat, flat_delim), 1).reshape(B, S, H * W * E + E)
    ind_delim = torch.tensor(delim_ind).to(device)  # add new cluster for delim
    ind_delim = ind_delim.repeat(B * S, 1)
    action_z_q_ind = actions + num_state_embeddings
    action_z_q_ind = action_z_q_ind.reshape(B * S, 1)
    z_q_ind = z_q_ind.reshape(B * S, TiF)
    z_q_ind = torch.cat((z_q_ind, action_z_q_ind, ind_delim), -1)
    z_q_ind = z_q_ind.reshape(B, S, tokens_in_frame)
    return z_q_ind
