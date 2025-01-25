import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoTokenizer

from einops import rearrange
from utils import *
from config import *
from modules import *


class VIT(nn.Module):
    def __init__(self, vit_config):
        super().__init__()
        self.config = vit_config
        self.patch_num = int(vit_config['image_size'] / vit_config['patch_size']) ** 2
        self.conv = nn.Conv2d(vit_config['in_channels'], vit_config['transformer_config']['d_model'], kernel_size=vit_config['patch_size'], stride=vit_config['patch_size'])
        self.cls_token = nn.Parameter(torch.rand(1, 1, vit_config['transformer_config']['d_model']))
        self.patch_embed = nn.Parameter(torch.rand(1, self.patch_num + 1, vit_config['transformer_config']['d_model']))
        self.encoder = Encoder(vit_config['transformer_config'])
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')

    def forward(self, x):
        if x.shape[2] != self.config['image_size'] or x.shape[3] != self.config['image_size']:
            raise RuntimeError(f'Input image shape {x.shape} must be identical to that in VIT_CONFIG {self.config['image_size']}!')
        if x.shape[1] != self.config['in_channels']:
            raise RuntimeError(f'Input image channel {x.shape[1]} does not match VIT_CONFIG {self.config['in_channels']}!')

        x = self.conv(x)
        x = rearrange(x, 'b l p1 p2 -> b (p1 p2) l')
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1) + self.patch_embed
        x = self.encoder(x, embed = False) # Embedding for image is done by this class, so we don't need to use sinusoidal to encode it again
        return x

class VITForClassification(nn.Module):
    def __init__(self, vit_config, num_classes = 10):
        super().__init__()
        self.vit = VIT(vit_config)
        self.classification_head = nn.Linear(vit_config['transformer_config']['d_model'], num_classes)
    
    def forward(self, x, target = None):
        x = self.vit(x)
        x = x[:, 0, :] # Select the first [cls] token
        logits = self.classification_head(x)
        
        if target is not None:
            loss = F.cross_entropy(logits, target)
            return logits, loss
        else:
            return logits

