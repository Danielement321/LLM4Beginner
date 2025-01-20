import torch

CONFIG = {
    'd_model': 768,
    'max_seq_length': 64,
    'num_heads': 12,
    'encoder_depth': 6, # This is ignored if you use DecoderOnly Setting.
    'decoder_depth': 12,
    'ffn_dim': 3072,
    'dropout': 0.1,
    'vocab_size': 999999,
    'eps': 1e-6,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tokenizer': 'google-bert/bert-base-uncased',
}

VIT_CONFIG = {
    'patch_size': 14,
    'image_size': 224,
    'in_channels': 3,
    'device': 'cuda',
    'transformer_config': CONFIG,
}

GENERATE_CONFIG = {
    'temperature': 1,
    'greedy': False,
}