import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.c_fc = nn.Linear(config.d_model, config.mlp_hidden_dim)
        
        if config.activation == "GELU":
            self.act = nn.GELU(approximate="tanh")
        elif config.activation == "ReLU":
            self.act = nn.ReLU()
        else:
            print(f"Invalid activation function: {config.activation}.")
        
        self.c_proj = nn.Linear(config.mlp_hidden_dim, config.d_model)
        self.c_proj.SCALE_INIT_FLAG = 1

    def forward(self, x):
        
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        
        return x