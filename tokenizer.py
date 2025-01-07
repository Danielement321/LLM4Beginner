import torch
from config import *

class Tokenizer():
    def __init__(self, lines):
        self.vocab_size = self.get_vocab_size(lines)
        self.encoder = {ch: i for i, ch in enumerate(sorted(set(lines)))}
        self.decoder = {i: ch for i, ch in enumerate(sorted(set(lines)))}
        CONFIG['vocab_size'] = self.vocab_size

    def get_vocab_size(self, lines):
        vocab_size = len(set(lines))
        return vocab_size
    
    def encode(self, x, return_pt = False):
        if return_pt:
            return torch.tensor([self.encoder[ch] for ch in x])
        return [self.encoder[ch] for ch in x]

    def decode(self, x):
        return [self.decoder[i] for i in x]
