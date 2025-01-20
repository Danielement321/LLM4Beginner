import torch
from config import *

def random_generate(model, tokenizer, batch_size = 5, max_new_tokens = 50):
    idx = torch.randint(0, config['vocab_size'], (batch_size, 1)).long().to(config['device'])

    model.eval()
    for length in range(1, max_new_tokens + 1):
        casual_mask = torch.triu(torch.ones([length, length]), diagonal=1).unsqueeze(0).to(config['device'])*(-1e9)
        logits = model(idx[:, -config['max_seq_length'] :], casual_mask = casual_mask)
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
    idx = tokenizer(context, return_tensors='pt')['input_ids'][:, 1: -1].to(config['device'])
    
    # idx = tokenizer(context, return_tensors='pt')['input_ids'].to(CONFIG['device'])
    idx = idx.repeat(batch_size, 1)

    model.eval()
    for length in range(idx.shape[1], max_new_tokens + idx.shape[1]):
        casual_mask = torch.triu(torch.ones([length, length]), diagonal=1).unsqueeze(0).to(config['device'])*(-1e9)
        logits = model(idx[:, -config['max_seq_length'] :], casual_mask = casual_mask)
        logits = torch.softmax(logits[:, -1, :] / GENERATE_CONFIG['temperature'], dim=-1)
        if GENERATE_CONFIG['greedy']:
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
        else:
            idx_next = torch.multinomial(logits, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    generated_text = [''.join(tokenizer.decode(x)) for x in idx.tolist()]

    return generated_text
