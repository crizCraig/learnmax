"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64, input_width=32, embedding_dim=64, is_single_token2=False,
                 strides=None):
        if strides is None:
            strides = [2, 2]
        super().__init__()

        self.is_single_token2 = is_single_token2

        down_sample = np.prod(1/np.array(strides))
        out_width = int(input_width * down_sample)

        if self.is_single_token2:
            self.output_channels = n_hid  # Want to see what decoding the encoder vs quantized looks like
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, n_hid, 4, stride=strides[0], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, 2 * n_hid, 4, stride=strides[1], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * n_hid, self.output_channels, 3, padding=1),
                nn.ReLU(),
                ResBlock(self.output_channels, 2 * n_hid // 4),
                ResBlock(self.output_channels, 2 * n_hid // 4),  # 128, 8x8
            )
        else:
            self.output_channels = 2 * n_hid
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, n_hid, 4, stride=strides[0], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_hid, 2*n_hid, 4, stride=strides[1], padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
                nn.ReLU(),
                ResBlock(2*n_hid, 2*n_hid//4),
                ResBlock(2*n_hid, 2*n_hid//4),  # 128, 8x8
            )

        self.out_width = out_width
        # self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, encoder, n_init=32, n_hid=64, output_channels=3, embedding_dim=64, is_single_token2=False,
                 strides=None):
        if strides is None:
            strides = [2, 2]
        super().__init__()
        self.is_single_token2 = is_single_token2
        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=strides[1], padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=strides[0], padding=1),
        )

    def forward(self, x):
        # for layer in self.net:
        #     x = layer.forward(x)
        # return x
        return self.net(x)
