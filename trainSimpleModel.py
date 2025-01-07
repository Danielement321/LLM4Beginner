import torch
from config import *
from data_utils import *
from models import *
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from generation import random_generate

lines = load_lines('data')
tokenizer = Tokenizer(lines)
lines = tokenizer.encode(lines, return_pt=True)
dataset = SimpleDatasetForCasualLM(lines, TRAIN_CONFIG['sample_size'], CONFIG)
dataloader = DataLoader(dataset, batch_size=TRAIN_CONFIG['train_batch'])

model = SimpleModel(CONFIG).to(CONFIG['device'])
optimizer = torch.optim.Adam(model.parameters())

losses = []
model.train()
for epoch in range(TRAIN_CONFIG['epochs']):
    for src, dst in tqdm(dataloader):
        src = src.to(CONFIG['device'])
        dst = dst.to(CONFIG['device'])
        logits, loss = model(src, dst)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.save(model.state_dict(), 'ckpts/SimpleModel.pth')

plt.plot(losses)
plt.title('Training Loss For Simple Model')
plt.savefig('losses.png')

print('Training Finished!')
print(random_generate(model, tokenizer))