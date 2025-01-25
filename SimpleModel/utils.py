import matplotlib.pyplot as plt
from config import *
import math
import datasets

class Colors:
    RED = '\033[1m\033[31m'
    GREEN = '\033[1m\033[32m'
    BLUE = '\033[1m\033[34m'
    YELLOW = '\033[1m\033[33m'
    MAGENTA = '\033[1m\033[35m'
    CYAN = '\033[1m\033[36m'
    RESET = '\033[1m\033[0m'

def config_check(config):
    print(f'CONFIG:{config}')
    if config.device == 'cpu':
        print(Colors.RED + 'CUDA is not availabel! Using CPU as the main device.' + Colors.RESET)
    if config.vocab_size == 999999:
        print(Colors.RED + 'The vocab_size is not set according to the size of tokenizer, this might cause OOM!' + Colors.RESET)
    if hasattr(config, 'num_attention_heads') and config.hidden_size % config.num_attention_heads != 0:
        raise RuntimeError('hidden_size % num_attention_heads must be 0!')
    if config.hidden_size % 2 != 0:
        raise ValueError('hidden_size must be even!')

def download_dataset(dataset_path, output_path):
    dataset = datasets.load_dataset(dataset_path, split='train')
    dataset = dataset.to_json(output_path)
    print('The dataset from hub has been saved as json files!')