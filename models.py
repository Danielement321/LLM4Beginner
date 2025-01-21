import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from einops import rearrange
from config import *
from modules import *


class DecoderOnlyTransformer(PreTrainedModel):
    config_class = SimpleDecoderOnlyTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.decoder = Decoder(self.config)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.attention_map = False
        self.mask = torch.triu(torch.ones([config.max_seq_len, config.max_seq_len]), diagonal=1).unsqueeze(0) * (-1e9)
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')
    
    def forward(self, input_ids, labels = None):
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        mask = self.mask.repeat(batch_size, 1, 1, 1).to(self.config.device)
        hidden_states = self.decoder(dst = input_ids, causal_mask = mask[:, :, :seq_len, :seq_len])
        logits = self.output(hidden_states)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
        else:
            return CausalLMOutput(logits=logits)
        
    def apply_attention_map(self):
        self.attention_map = True
        for block in self.decoder.decoder_blocks:
            if isinstance(block, DecoderBlock):
                block.self_attention = MultiHeadSelfAttentionWithMap(self.config)
                block.to(self.config.device)
    
    @torch.inference_mode
    def generate(self, idx, max_new_tokens = 50, temperature = 1):
        self.eval()
        for length in range(idx.shape[1], max_new_tokens + idx.shape[1]):
            logits = self(idx[:, -self.config.max_seq_len :])['logits']
            logits = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            if temperature < 1e-3:
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
            else:
                idx_next = torch.multinomial(logits, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

    @torch.inference_mode
    def random_generate(self, batch_size = 5, max_new_tokens = 50):
        idx = torch.randint(0, self.config.vocab_size, (batch_size, 1)).long().to(self.config.device)
        idx = self.generate(idx, max_new_tokens)
        return idx

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

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['ffn_dim']),
            nn.ReLU(),
            nn.Linear(config['ffn_dim'], config['d_model'])
        )
        self.out = nn.Linear(config['d_model'], config['vocab_size'])
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')

    def forward(self, src, dst = None, causal_mask = None): # causal_mask is not used by this very simple model, we add causal_mask here to unify the function arguments for generation
        x = self.embedding(src)
        x = self.linear(x)
        logits = self.out(x)

        if dst is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), dst.view(-1))
            return logits, loss
        
        else:
            return logits