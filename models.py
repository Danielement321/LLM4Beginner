import torch
import torch.nn as nn
from torch.nn import functional as F

from config import *
from config import CONFIG

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config['d_model'], config['vocab_size'])
    
    def forward(self, src, dst, padding_mask = None, casual_mask = None):
        src = self.encoder(src, padding_mask)
        decoded = self.decoder(src, dst, padding_mask, casual_mask)
        logits = self.linear(decoded)
        return logits


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config['d_model'], config['vocab_size'])
    
    def forward(self, x, casual_mask):
        decoded = self.decoder(x, casual_mask = casual_mask)
        logits = self.linear(decoded)
        return logits

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['ffn_dim']),
            nn.ReLU(),
            nn.Linear(config['ffn_dim'], config['d_model'])
        )
        self.out = nn.Linear(config['d_model'], config['vocab_size'])

    def forward(self, src, dst = None):
        x = self.embedding(src)
        x = self.linear(x)
        logits = self.out(x)

        if dst is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), dst.view(-1))
            return logits, loss
        
        else:
            return logits