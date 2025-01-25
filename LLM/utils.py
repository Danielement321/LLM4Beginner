import matplotlib.pyplot as plt
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

def download_dataset(dataset_path, output_path):
    dataset = datasets.load_dataset(dataset_path, split='train')
    dataset = dataset.to_json(output_path)
    print('The dataset from hub has been saved as json files!')