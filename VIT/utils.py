import os
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from config import *

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


class IdenticalDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform):
        self.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transform
        self._validate_data(max_validation_workers=16)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = self.transform(image)
        return image, image
    
    def _validate_data(self, max_validation_workers):
        print("Validating Data...")
        valid_samples = []

        with ThreadPoolExecutor(max_workers=max_validation_workers) as executor:
            futures = {
                executor.submit(self._validate_single_data, sample): sample
                for sample in self.images
            }

            for future in tqdm(as_completed(futures), total=len(self.images)):
                sample = futures[future]
                is_valid = future.result()
                if is_valid:
                    valid_samples.append(sample)

        self.images = valid_samples
        self.num = len(self.images)
        print(Colors.GREEN + f'Validated Data: {self.num} samples' + Colors.RESET)
    
    def _validate_single_data(self, sample):
        try:
            Image.open(sample).close()
            return True
        except Exception as e:
            print(Colors.YELLOW + f"Error in file {sample}: {e}" + Colors.RESET)
            return False


class IdenticalCollator:
    def __init__(self):
        pass

    def __call__(self, x):
        images = torch.stack([x[i][0] for i in range(len(x))])
        targets = torch.stack([x[i][1] for i in range(len(x))])
        return {'pixel_values': images, 'labels': targets}