import logging

# make deterministic
from constants import IS_DEBUG_MODE, BLOCK_SIZE, SAVE_DIR
from mingpt.utils import set_seed

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# alright, let's sample some character-level Shakespeare
from mingpt.utils import sample


set_seed(42)
class CharDataset(Dataset):

    def __init__(self, data, block_size, percent_random=0, all_same=False):  # block_size=128
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)  # vocab_size=65, data_size=1115390
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.percent_random = percent_random
        self.all_same = all_same

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size
        random_chunk = []
        if self.percent_random:
            random_size = block_size * self.percent_random // 100
            block_size = block_size - random_size
            random_chunk = np.random.randint(low=0, high=self.vocab_size, size=(random_size,)).tolist()

        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk] + random_chunk
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during training will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can parallelize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.

        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """

        if self.all_same:
            dix = [1] * len(dix)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def main():
    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    train_dataset = CharDataset(open('train_shakespeare.txt', 'r').read(), BLOCK_SIZE)  # one line of poem is roughly 50 characters
    test_dataset = CharDataset(open('test_shakespeare.txt', 'r').read(), BLOCK_SIZE)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    if IS_DEBUG_MODE:
        num_workers = 0
    else:
        num_workers = 1

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=2, batch_size=64, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512 * 20, final_tokens=2 * len(train_dataset) * BLOCK_SIZE,
                          num_workers=num_workers, ckpt_path=SAVE_DIR)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()


# torch.cuda.empty_cache()
if __name__ == '__main__':
    main()



"""
epoch 1 iter 7632: train loss 0.54597. lr 5.317630e-04:  44%|████▍     | 7633/17426 [28:24<36:45,  4.44it/s]

Game plan: Add vocab_size * 2 outputs for prob and variance. Then add a term to the loss similar to what PPO does with the gausssian

Looks like sac kind of does what we want with

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
log_std in PPO however is not a function of input.

Certainty loss will be
    target = 1 or 0 if predicted or not given some input z(s,a) [We should look at gap between quantized and latent to 
        see if too large - perhaps then update/fine-tune AE (end-to-end?) or don't mark as predicted]
    output = transformer logit output for certainty across all quantized z 
    
Check initial variance values and make sure they are reasonable. Looks reasonable! Goes to 0 with random, goes to 0.123 with 65 output chars, and climbs from 0.4-0.8 during training on Shakespeare (real data).
https://app.neptune.ai/crizcraig/safeobjective/experiments?compare=IwBgNArGAslA&split=cmp&dash=charts&viewId=standard-view

Check deviation head with random and repeated data. Makes sense - deviation 

"""
