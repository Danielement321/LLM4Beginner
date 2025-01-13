import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import *
from data_utils import *
from utils import *
from generate_utils import random_generate
from models import DecoderOnlyTransformer
from torch.utils.tensorboard import SummaryWriter

epochs = 1
lr = 1e-3
train_batch = 96
sample_size = 2000

writer = SummaryWriter('runs')
tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer'])
CONFIG['vocab_size'] = tokenizer.vocab_size

lines = load_lines('data/*.txt')
tokenized_lines = tokenizer(lines[:20000], return_tensors='pt')
dataset = DatasetForCasualLM(tokenized_lines, num=sample_size, config=CONFIG)
dataloader = DataLoader(dataset, train_batch, shuffle=True)
steps = len(dataset)/train_batch*epochs

config_check()
model = DecoderOnlyTransformer(CONFIG).to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

model.train()
for epoch in range(epochs):
    for step, (src_input_ids, src_casual_mask, dst_input_ids) in enumerate(tqdm(dataloader)):
        src_input_ids = src_input_ids.to(CONFIG['device'])
        src_casual_mask = src_casual_mask.to(CONFIG['device'])
        dst_input_ids = dst_input_ids.to(CONFIG['device'])

        logits, loss = model(src_input_ids, src_casual_mask, dst_input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tqdm.write(f'loss:{loss.item()}, lr:{scheduler.get_last_lr()[0]}')
        writer.add_scalar('loss', loss.item(), (epoch + 1) * step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], (epoch + 1) * step)

    torch.save(model.state_dict(), 'ckpts/DecoderOnlyTransformer.pth')

print('Training Finished!')
print(random_generate(model, tokenizer))