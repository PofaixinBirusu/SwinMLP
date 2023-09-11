import torch
from torch import nn
from torch.nn import functional as F


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)  # 偏差

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)  # (256, d_ffn * 2=1024)  [-1,256,1024]
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)  #
        self.channel_proj2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        # batch x seq_len x d_model
        residual = x
        x = self.norm(x)  # [-1,256,256]
        x = F.gelu(self.channel_proj1(x))  # GELU激活函数 [-1,256,256]
        x = self.sgu(x)  # [-1,256,256]
        x = self.channel_proj2(x)
        out = x + residual
        return out