import torch
from config import *

def generate(model, tokenizer, max_new_tokens = 50):
    idx = torch.randint(0, CONFIG['vocab_size'], (5, 1)).long().to(CONFIG['device'])

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
