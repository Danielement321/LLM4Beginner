import torch
from config import *
from data_utils import *
from models import *
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from generation import generate

lines = open('data/xiyouji.txt', 'r').read()
tokenizer = Tokenizer(lines)
lines = tokenizer.encode(lines, return_pt=True)
dataset = SimpleDatasetForCasualLM(lines, 50000, CONFIG)
dataloader = DataLoader(dataset, batch_size=TRAIN_CONFIG['train_batch'])

model = SimpleModel(CONFIG).to(CONFIG['device'])
optimizer = torch.optim.Adam(model.parameters())

losses = []
model.train()
for epoch in range(TRAIN_CONFIG['epochs']):
    for src, dst in dataloader:
        src = src.to(CONFIG['device'])
        dst = dst.to(CONFIG['device'])
        logits, loss = model(src, dst)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

print(generate(model, tokenizer))
torch.save(model.state_dict(), 'ckpts/model.pth')
plt.plot(losses)
plt.savefig('losses.png')