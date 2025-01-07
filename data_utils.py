import torch
from torch.utils.data import Dataset
import random
import os

class SimpleDatasetForCasualLM(Dataset):
    def __init__(self, tokenized_lines, num, config, lenghth_noise = 5):
        self.num = num
        self.max_seq_len = config['max_seq_length']
        self.tokenized_lines = tokenized_lines.squeeze()
        self.total_seq_length = self.tokenized_lines.shape[0]
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

class DatasetForCasualLM(Dataset):
    def __init__(self, tokenized, num, config):
        self.num = num
        self.max_seq_len = config['max_seq_length']
        self.tokenized = tokenized['input_ids'].squeeze()
        self.total_seq_length = self.tokenized.shape[0]
        self.src_causal_mask = torch.triu(torch.ones([self.max_seq_len, self.max_seq_len]), diagonal=1).unsqueeze(0) * (-1e9)
        assert self.total_seq_length - config['max_seq_length'] >= num
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        start = random.randint(0, self.total_seq_length - self.max_seq_len - idx -2)
        src_input_ids = self.tokenized[start + idx: start + idx + self.max_seq_len]
        dst_input_ids = self.tokenized[start + idx + 1: start + idx + self.max_seq_len + 1]
        return src_input_ids, self.src_causal_mask, dst_input_ids

def load_lines(folder_path) -> str:
    lines = ''
    for file in os.listdir(folder_path):
        try:
            with open(os.path.join(folder_path, file), 'r') as f:
                lines += f.read()
        except:
            print(f'Error while reading {file}')
    return lines