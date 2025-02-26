import torch
from transformers import PretrainedConfig


class SimpleVITConfig(PretrainedConfig):
    model_type = 'simple_vit'
    
    def __init__(self,
                 hidden_size = 768,
                 num_attention_heads = 12,
                 n_layers = 6,
                 intermediate_size = 3072,
                 dropout = 0.1,
                 patch_size = 14,
                 image_size = 224,
                 in_channels = 3,
                 flash_attn = True,
                 eps = 1e-6,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.n_layers = n_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.flash_attn = flash_attn
        self.eps = eps
        super().__init__(**kwargs)