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
import matplotlib.pyplot as plt
from generate_utils import random_generate
from models import SimpleModel

epochs = 1
lr = 1e-3
train_batch = 96
sample_size = 2000

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
config['vocab_size'] = tokenizer.vocab_size

lines = load_lines('data/*.txt')
tokenized_lines = tokenizer(lines[:20000], return_tensors='pt')
dataset = DatasetForCasualLM(tokenized_lines, num=sample_size, config=config)
dataloader = DataLoader(dataset, train_batch, shuffle=True)

config_check()
model = SimpleModel(config).to(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []

model.train()
for epoch in range(epochs):
    for src_input_ids, _, dst_input_ids in tqdm(dataloader):
        src_input_ids = src_input_ids.to(config['device'])
        dst_input_ids = dst_input_ids.to(config['device'])

        logits, loss = model(src_input_ids, dst_input_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.save(model.state_dict(), 'ckpts/SimpleModel.pth')

plt.plot(losses)
plt.title('Training Loss SimpleModel')
plt.savefig('losses.png')

print('Training Finished!')

idx = tokenizer('the government', return_tensors='pt')['input_ids'][:, 1: -1].to(self.config.device)
print(random_generate(model, tokenizer))