import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutput

from einops import rearrange
from utils import *
from config import *
from modules import *


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


def load_model(model_path: str):
    try:
        if model_path.endswith('.pth'): # DecoderOnlyTransformer trained with trainDecoder.py
            tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
            config = torch.load(model_path, weights_only=False)['config']
            config_check(config)
            model = DecoderOnlyTransformer(config)
            model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
            return tokenizer, model.to(config.device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = DecoderOnlyTransformer.from_pretrained(model_path)
            config_check(model.config)
            return tokenizer, model.to(model.config.device)
    except:
        raise ValueError('Unrecognized model! If you are using a model trained with transformers trainer, please pass absolute path.')
