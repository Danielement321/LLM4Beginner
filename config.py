CONFIG = {
    'd_model': 128,
    'max_seq_length': 16,
    'ffn_dim': 2048,
    'dropout': 0.1,
    'vocab_size': 999999,
    'device': 'cuda',
    '-inf': -1e9,
}

TRAIN_CONFIG = {
    'train_batch': 64,
    'lr': 0.001,
    'epochs': 3,
    'eval_size': 0.1,
    'eval_batch': 96,
}

GENERATE_CONFIG = {
    'temperature': 1,
    'greedy': False,
}