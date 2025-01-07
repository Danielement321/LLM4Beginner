import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models import DecoderOnlyTransformer
from config import *
from tqdm import tqdm
from generation import generate
from data_utils import DatasetForCasualLM
from utils import *
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
lines = load_lines('data')
CONFIG['vocab_size'] = tokenizer.vocab_size

tokenized_lines = tokenizer(lines, return_tensors='pt')
dataset = DatasetForCasualLM(tokenized_lines, num=20000, config=CONFIG)
dataloader = DataLoader(dataset, TRAIN_CONFIG['train_batch'], shuffle=True)

config_check()
model = DecoderOnlyTransformer(CONFIG).to(CONFIG['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['lr'])

losses = []

model.train()
for epoch in range(TRAIN_CONFIG['epochs']):
    for src_input_ids, src_casual_mask, dst_input_ids in tqdm(dataloader):
        src_input_ids = src_input_ids.to(CONFIG['device'])
        src_casual_mask = src_casual_mask.to(CONFIG['device'])
        dst_input_ids = dst_input_ids.to(CONFIG['device'])

        logits, loss = model(src_input_ids, src_casual_mask, dst_input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

torch.save(model.state_dict(), 'ckpts/model.pth')

plt.plot(losses)
plt.savefig('losses.png')

print(generate(model, tokenizer))