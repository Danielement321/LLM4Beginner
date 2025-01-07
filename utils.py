import torch
import warnings
import einops
from config import *
from tqdm import tqdm
import os

def config_check():
    print(f'CONFIG:{CONFIG}')
    if CONFIG['device'] == 'cpu':
        warnings.warn('Using CPU as the main device! This can be changed in config.py')
    if CONFIG['vocab_size'] == 999999:
        warnings.warn('The vocab_size is not set according to the size of tokenizer, this might cause errors or OOM!')
    if CONFIG['d_model'] % CONFIG['num_heads'] != 0:
        raise RuntimeError('d_model % num_heads must be 0!')
