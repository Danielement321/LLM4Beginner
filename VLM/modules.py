import torch
import torch.nn as nn
from config import SimpleVLMConfig

class Projector(nn.Module):
    def __init__(self, config: SimpleVLMConfig):
        super().__init__()
        self.config = config
        self.ffn = nn.Sequential(
            nn.Linear(config.vision_tower_hidden_size, config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size)
        )
    
    def forward(self, x):
        return self.ffn(x)