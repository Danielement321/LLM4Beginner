import matplotlib.pyplot as plt
import warnings
from config import *
from tqdm import tqdm
import math
import os

def config_check():
    print(f'CONFIG:{CONFIG}')
    if CONFIG['device'] == 'cpu':
        warnings.warn('Using CPU as the main device! This can be changed in config.py')
    if CONFIG['vocab_size'] == 999999:
        warnings.warn('The vocab_size is not set according to the size of tokenizer, this might cause errors or OOM!')
    if CONFIG['d_model'] % CONFIG['num_heads'] != 0:
        raise RuntimeError('d_model % num_heads must be 0!')
    if GENERATE_CONFIG['temperature'] <=0 :
        raise RuntimeError('temperature in GENERATION_CONFIG must > 0!')

def vit_config_check():
    print(f'VIT CONFIG:{VIT_CONFIG}')
    config_check()
    if VIT_CONFIG['image_size'] % VIT_CONFIG['patch_size'] != 0:
        raise RuntimeError(f'image_size % patch_size must be 0!')

def plot_attention(model, layer = 0, batch_idx = 0):
    if not hasattr(model, 'attention_map') or not model.attention_map:
        raise RuntimeError('Current model does not support attention_map, please run `model.apply_attention_map()` first!')
    model.eval()
    atten = model.decoder.decoder_blocks[layer].self_attention.attention_weights[batch_idx].cpu().detach().numpy()
    num_heads = atten.shape[0]

    height = int(math.sqrt(num_heads))
    width = math.ceil(num_heads / height)
    fig, axs = plt.subplots(height, width, figsize=(15, 10))
    for i in range(num_heads):
        ax = axs[i // width, i % width]
        cax = ax.matshow(atten[i], cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.axis('off')
    
    fig.suptitle(f'Attention Map For Layer{layer}')
    plt.tight_layout()
    plt.show()