CONFIG = {
    'd_model': 768,
    'max_seq_length': 64,
    'num_heads': 12,
    'decoder_depth': 12,
    'ffn_dim': 3072,
    'dropout': 0.1,
    'vocab_size': 999999,
    'device': 'cuda',
    'tokenizer': 'google-bert/bert-base-uncased',
}

TRAIN_CONFIG = {
    'train_batch': 96,
    'lr': 0.001,
    'epochs': 2,
    'eval_size': 0.1,
    'eval_batch': 96,
}

GENERATE_CONFIG = {
    'temperature': 1,
    'greedy': False,
}