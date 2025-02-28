import matplotlib.pyplot as plt
from config import *
import math
from sklearn.metrics import accuracy_score
import torch

class Colors:
    RED = '\033[1m\033[31m'
    GREEN = '\033[1m\033[32m'
    BLUE = '\033[1m\033[34m'
    YELLOW = '\033[1m\033[33m'
    MAGENTA = '\033[1m\033[35m'
    CYAN = '\033[1m\033[36m'
    RESET = '\033[1m\033[0m'


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

    
class ImageClassificationCollator:
    def __init__(self):
        pass

    def __call__(self, x):
        images, targets = zip(*x)
        images = torch.stack(images)
        targets = torch.tensor(targets)
        return {'pixel_values': images, 'labels': targets}

def compute_metrics(p):
    preds, labels = p
    preds = torch.argmax(torch.tensor(preds), dim=1)
    accuracy = accuracy_score(labels, preds)
    print(Colors.BLUE + f'\nAccuracy: {accuracy:.4f}' + Colors.RESET)
    return {"accuracy": accuracy}