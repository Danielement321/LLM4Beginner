import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class Colors:
    RED = '\033[1m\033[31m'
    GREEN = '\033[1m\033[32m'
    BLUE = '\033[1m\033[34m'
    YELLOW = '\033[1m\033[33m'
    MAGENTA = '\033[1m\033[35m'
    CYAN = '\033[1m\033[36m'
    RESET = '\033[1m\033[0m'
