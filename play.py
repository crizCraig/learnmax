import numpy as np
from torch import nn
import torch

mlp = nn.Linear(3, 2, bias=False)
B = 2
x = torch.tensor(np.arange(B * 3).reshape(B, 3).astype(np.float32))
y = mlp(x)

print(y)
