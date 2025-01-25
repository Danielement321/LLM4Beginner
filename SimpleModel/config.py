import torch
from transformers import PretrainedConfig


class SimpleModelConfig(PretrainedConfig):
    model_type = 'simple_ffn_model'

    def __init__(self, 
                hidden_size = 768,
                max_seq_len = 64,
                n_layers = 12,
                intermediate_size = 3072,
                dropout = 0.1,
                vocab_size = 999999,
                eps = 1e-6,
                **kwargs):
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eps = eps
        super().__init__(**kwargs)
