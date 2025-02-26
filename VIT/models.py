import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from einops import rearrange
from utils import *
from config import *
from modules import *


class VIT(PreTrainedModel):
    config_class = SimpleVITConfig
    
    def __init__(self, config: SimpleVITConfig):
        super().__init__(config)
        self.config = config
        self.patch_num = int(config.image_size / config.patch_size) ** 2
        self.conv = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, config.hidden_size))
        self.patch_embed = nn.Parameter(torch.rand(1, self.patch_num + 1, config.hidden_size))
        self.encoder = Encoder(config)
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')

    def forward(self, x):
        if x.shape[2] != self.config.image_size or x.shape[3] != self.config.image_size:
            raise RuntimeError(f'Input image shape {x.shape} must be identical to that in VIT_CONFIG {self.config.image_size}!')
        if x.shape[1] != self.config.in_channels:
            raise RuntimeError(f'Input image channel {x.shape[1]} does not match VIT_CONFIG {self.config.in_channels}!')

        x = self.conv(x)
        x = rearrange(x, 'b l p1 p2 -> b (p1 p2) l')
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1) + self.patch_embed
        x = self.encoder(x, embed = False)
        return x

class VITForClassification(VIT):
    def __init__(self, config, num_classes = 10):
        super().__init__(config)
        self.vit = VIT(config)
        self.classification_head = nn.Linear(config.hidden_size, num_classes)
    
    def forward(self, pixel_values, labels = None):
        pixel_values = self.vit(pixel_values)
        pixel_values = pixel_values[:, 0, :] # Select the first [cls] token
        logits = self.classification_head(pixel_values)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        else:
            return ImageClassifierOutput(logits=logits)