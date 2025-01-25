import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from einops import rearrange
from utils import *
from config import *
from modules import *

class SimpleModel(PreTrainedModel):
    config_class = SimpleModelConfig
    
    def __init__(self, config :SimpleModelConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Sequential(*[FFN(config) for i in range(config.n_layers)])
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')

    def forward(self, input_ids, labels = None):
        x = self.embedding(input_ids)
        x = self.linear(x)
        logits = self.output(x)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return CausalLMOutput(loss=loss, logits=logits)
        
        else:
            return CausalLMOutput(logits=logits)
        
    @torch.inference_mode
    def generate(self, idx, max_new_tokens = 50, temperature = 1):
        self.eval()
        for length in range(idx.shape[1], max_new_tokens + idx.shape[1]):
            logits = self(idx[:, -self.config.max_seq_len :])['logits']
            logits = logits[:, -1, :]
            if temperature < 1e-2:
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
            else:
                logits = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(logits, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx

    @torch.inference_mode
    def random_generate(self, batch_size = 5, max_new_tokens = 50):
        idx = torch.randint(0, self.config.vocab_size, (batch_size, 1)).long().to(self.config.device)
        idx = self.generate(idx, max_new_tokens)
        return idx

class DecoderOnlyTransformer(PreTrainedModel):
    config_class = SimpleDecoderOnlyTransformerConfig

    def __init__(self, config: SimpleDecoderOnlyTransformerConfig):
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
                original_weights = block.self_attention.state_dict()
                block.self_attention = MultiHeadSelfAttentionWithMap(self.config).to(self.config.device)
                block.self_attention.load_state_dict(original_weights)
        print(Colors.MAGENTA + 'Attention_map is now supported! This may cause unnecessary memory consumption if you are not conducting a visualization.' + Colors.RESET)
    
    @torch.inference_mode
    def generate(self, idx, max_new_tokens = 50, temperature = 1):
        self.eval()
        for length in range(idx.shape[1], max_new_tokens + idx.shape[1]):
            logits = self(idx[:, -self.config.max_seq_len :])['logits']
            logits = logits[:, -1, :]
            if temperature < 1e-2:
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
            else:
                logits = torch.softmax(logits / temperature, dim=-1)
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

def load_model(model_path, load_mode = 'trainer'):
    if load_mode == 'trainer': # DecoderOnlyTransformer trained with trainWithTrainer.py
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = DecoderOnlyTransformer.from_pretrained(model_path)
        config_check(model.config)
        return tokenizer, model.to(model.config.device)
    elif load_mode == 'custom_trainer': # DecoderOnlyTransformer trained with trainDecoder.py
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '<e>'})

        config = torch.load(model_path, weights_only=False)['config']
        config_check(config)
        model = DecoderOnlyTransformer(config).to(config.device)
        model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
        return tokenizer, model
    else: # The SimpleModel
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '<e>'})
        
        config = torch.load(model_path, weights_only=False)['config']
        config_check(config)
        model = SimpleModel(config).to(config.device)
        model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
        return tokenizer, model
