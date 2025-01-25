import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

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

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
    
    config = torch.load(model_path, weights_only=False)['config']
    config_check(config)
    model = SimpleModel(config).to(config.device)
    model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
    return tokenizer, model
