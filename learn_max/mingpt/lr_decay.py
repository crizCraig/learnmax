import math

import pytorch_lightning as pl

from learn_max import dvq
from learn_max.utils import get_batch_vars


class GptWarmupCosineLearningRateDecay(pl.Callback):
    """
    based on the number of tokens seen during training will adjust the learning rate:
    1. first it will start at zero and gradually ramp up to full learning rate
    2. then it will decay down with the cosine learning rate decay down until 10% of original
    """

    def __init__(self, learning_rate, warmup_tokens, final_tokens):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        # state in this class, will count number of tokens processed so far
        self.tokens = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx=None):
        # y needs to be target indexes which is currently not passed by train_batch, do so when training_gpt as we have them! Also, need to emulate -100 thing when stars align
        gpt_x, z_q_ind_x, z_q_ind_y, a, s = get_batch_vars(batch, return_agent_state=True, populate_gpt=True)
        # _, y = batch
        self.tokens += (z_q_ind_y >= 0).sum()  # y == -100 is "ignore", so don't count these
        if self.tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
        else:
            # followed by cosine learning rate decay
            progress = float(self.tokens - self.warmup_tokens) / float(
                max(1, self.final_tokens - self.warmup_tokens))
            lr_mult = 0.1 + 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.learning_rate * lr_mult
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
