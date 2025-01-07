import torch
from torch.utils.data import Dataset
import random

class DatasetForCasualLM(Dataset):
    def __init__(self, tokenized_lines, num, config, lenghth_noise = 5):
        self.num = num
        self.max_seq_len = config['max_seq_length']
        self.tokenized_lines = tokenized_lines
        self.total_seq_length = tokenized_lines.shape[0]
        self.length_noise = lenghth_noise
        assert lenghth_noise < self.max_seq_len
        assert self.total_seq_length - config['max_seq_length'] >= num
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        start = random.randint(0, self.total_seq_length - self.max_seq_len - idx -2)
        length_noise = random.randint(0, self.length_noise)
        src = self.tokenized_lines[start + idx - length_noise: start + idx + self.max_seq_len - length_noise]
        dst = self.tokenized_lines[start + idx + 1 - length_noise: start + idx + self.max_seq_len + 1 - length_noise]
        return src, dst