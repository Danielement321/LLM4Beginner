import torch
import warnings
import einops
from config import *
from tqdm import tqdm

def get_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask - 1
    mask = einops.rearrange(mask, 'b l -> b 1 l 1')
    mask = mask.int() * CONFIG['-inf']
    return mask

def get_casual_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.shape[1]
    tri_mask = torch.triu(torch.ones([seq_len, seq_len]).bool(), diagonal=1)
    pad_mask = get_padding_mask(attention_mask).bool()
    cas_mask = (tri_mask + pad_mask).int() * CONFIG['-inf']
    return cas_mask

def config_check():
    if CONFIG['device'] == 'cpu':
        warnings.warn('Using CPU as the main device! This can be changed in config.py')
    if CONFIG['vocab_size'] == 999999:
        warnings.warn('The vocab_size is not set according to the size of tokenizer, this might cause errors or OOM!')
    if CONFIG['d_model'] % CONFIG['num_heads'] != 0:
        raise RuntimeError('d_model % num_heads must be 0!')

def do_eval(model, eval_dataloader, tokenizer):
    model.eval()
    correct, total = 0, 0
    for tokenized_src, tokenized_dst in tqdm(eval_dataloader, desc=f"Evaluating"):
        english_input_ids = tokenized_src['input_ids'].to(CONFIG['device'])
        english_padding_mask = tokenized_src['padding_mask'].to(CONFIG['device'])
        chinese_input_ids = tokenized_dst['input_ids'].to(CONFIG['device'])
        chinese_casual_mask = tokenized_dst['casual_mask'].to(CONFIG['device'])
        valid_token_mask = chinese_input_ids.view(-1) != tokenizer.pad_token_id
        
        with torch.no_grad():
            logits = model(english_input_ids, chinese_input_ids, english_padding_mask, chinese_casual_mask)
            ans = torch.argmax(logits.view(-1, CONFIG['vocab_size']), dim=-1)
            correct += ((ans == chinese_input_ids.view(-1)) * valid_token_mask).sum().item()
            total += valid_token_mask.sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")