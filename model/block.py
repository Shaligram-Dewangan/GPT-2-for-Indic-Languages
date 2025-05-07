import torch
import torch.nn as nn

from .mlp import MLP
from .attention import CasualSelfAttention


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x