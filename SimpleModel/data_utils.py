import torch
from torch.utils.data import Dataset, IterableDataset
from glob import glob
from tqdm import tqdm
import json

class DatasetForCasualLM(Dataset):
    "This class reads a folder of txt files and convert them to context format."
    def __init__(self, tokenizer, data_folder, num, config):
        self.num = int(num)
        self.max_seq_len = config.max_seq_len
        self.tokenizer = tokenizer
        self.data = self.load_lines(data_folder)
        self.total_seq_length = len(self.data)
        if num > self.calcute_token_nums():
            raise ValueError('num should be less than total token numbers')
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        tokenized = self.tokenizer.encode(self.data[idx: idx + self.max_seq_len * 10])
        src_input_ids = tokenized[0: self.max_seq_len]
        dst_input_ids = tokenized[1: self.max_seq_len + 1]
        if len(dst_input_ids) < self.max_seq_len:
            dst_input_ids += [self.tokenizer.pad_token_id]
        return {'input_ids': torch.tensor(src_input_ids), 'labels': torch.tensor(dst_input_ids)}
    
    def load_lines(self, folder_path) -> str:
        lines = ''
        print("Reading Data...")
        for file in tqdm(glob(folder_path)):
            try:
                with open(file, 'r') as f:
                    lines += f.read()
            except Exception as e:
                print(f'{e} while reading {file}')
        if len(lines) == 0:
            raise RuntimeError('Length of training data is 0!')
        return lines
    
    def calcute_token_nums(self):
        split_words = self.data.split()
        print(f'Total words: {len(split_words):,}')
        return len(split_words)
