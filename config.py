CONFIG = {
    'd_model': 512,
    'max_seq_length': 64,
    'num_heads': 8,
    'decoder_depth': 12,
    'ffn_dim': 2048,
    'dropout': 0.1,
    'vocab_size': 999999,
    'device': 'cuda',
}

TRAIN_CONFIG = {
    'sample_size': 1500000,
    'train_batch': 128,
    'lr': 0.001,
    'epochs': 3,
    'eval_size': 0.1,
    'eval_batch': 96,
}

GENERATE_CONFIG = {
    'temperature': 1,
    'greedy': False,
}