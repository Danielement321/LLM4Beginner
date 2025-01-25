import torch
from transformers import PretrainedConfig, AutoConfig


class SimpleDecoderOnlyTransformerConfig(PretrainedConfig):
    model_type = 'simple_decoder_only_transformer'

    def __init__(self, 
                hidden_size = 768,
                num_attention_heads = 12,
                max_seq_len = 64,
                n_layers = 12,
                intermediate_size = 3072,
                dropout = 0.1,
                vocab_size = 999999,
                eps = 1e-6,
                flash_attn = False,
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eps = eps
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
