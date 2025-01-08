import torch
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
import random
import os

class DatasetForCasualLM(Dataset):
    def __init__(self, tokenized, num, config):
        self.num = int(num)
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
    print("Reading Data...")
    for file in tqdm(glob(folder_path)):
        try:
            with open(file, 'r') as f:
                lines += f.read()
        except:
            print(f'Error while reading {file}')
    return lines