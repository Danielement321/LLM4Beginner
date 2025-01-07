import torch
import warnings
import einops
from config import *
from tqdm import tqdm
import os

def load_lines(folder_path) -> str:
    lines = ''
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            lines += f.read()
    return lines

def config_check():
    print(f'CONFIG:{CONFIG}')
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