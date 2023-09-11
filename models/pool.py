import numpy as np
import torch
from torch import nn


class AttentivePool(nn.Module):
    def __init__(self, dim):
        super(AttentivePool, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.contiguous()
        # batch x n x c
        batch, n, c = x.shape[0], x.shape[1], x.shape[2]
        score = self.softmax(self.linear(x.view(batch*n, c)).view(batch, n, c) / np.sqrt(768))
        # batch x n x c, batch x c
        return score, torch.sum(x*score, dim=1)