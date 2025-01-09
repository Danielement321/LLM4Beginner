import torch
from config import *

def random_generate(model, tokenizer, batch_size = 5, max_new_tokens = 50):
    idx = torch.randint(0, CONFIG['vocab_size'], (batch_size, 1)).long().to(CONFIG['device'])

    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -CONFIG['max_seq_length'] :])
        logits = torch.softmax(logits[:, -1, :] / GENERATE_CONFIG['temperature'], dim=-1)
        if GENERATE_CONFIG['greedy']:
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
        else:
            idx_next = torch.multinomial(logits, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    generated_text = [''.join(tokenizer.decode(x)) for x in idx.tolist()]

    return generated_text

def context_generate(context: str, model, tokenizer, batch_size = 5, max_new_tokens = 50):
    # If you are using bert tokenizer, remove the [cls] and [sep] tokens
    idx = tokenizer(context, return_tensors='pt')['input_ids'][:, 1: -1].to(CONFIG['device'])
    
    # idx = tokenizer(context, return_tensors='pt')['input_ids'].to(CONFIG['device'])
    idx = idx.repeat(batch_size, 1)
    if idx.shape[1] >= max_new_tokens:
        raise RuntimeError(f'max_new_tokens:{max_new_tokens} must > context length {idx.shape[1]}')

    model.eval()
    for _ in range(idx.shape[1], max_new_tokens):
        logits = model(idx[:, -CONFIG['max_seq_length'] :])
        logits = torch.softmax(logits[:, -1, :] / GENERATE_CONFIG['temperature'], dim=-1)
        if GENERATE_CONFIG['greedy']:
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
        else:
            idx_next = torch.multinomial(logits, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    generated_text = [''.join(tokenizer.decode(x)) for x in idx.tolist()]

    return generated_text