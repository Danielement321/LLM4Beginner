import torch
import torch.nn as nn
from einops import rearrange


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.silu = nn.SiLU()
        self.norm = RMSNorm(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_ = x.clone()
        x = self.norm(x)
        gate = self.silu(self.linear1(x))
        x = self.linear2(x) * gate
        x = self.linear3(x)
        x = self.dropout(x)
        return x + x_


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * self.weight
        return x
