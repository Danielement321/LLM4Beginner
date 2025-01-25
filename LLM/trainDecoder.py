import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import SimpleDecoderOnlyTransformerConfig
from data_utils import DatasetForCasualLM
from utils import *
from models import DecoderOnlyTransformer
from torch.utils.tensorboard import SummaryWriter

epochs = 1
lr = 8e-4
train_batch = 64
sample_size = 200000

writer = SummaryWriter('runs')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
config = SimpleDecoderOnlyTransformerConfig(vocab_size=tokenizer.vocab_size)

dataset = DatasetForCasualLM(tokenizer, 'data/*.txt', num=sample_size, config=config)
dataloader = DataLoader(dataset, train_batch, shuffle=True)
steps = len(dataset)/train_batch*epochs

config_check(config)
model = DecoderOnlyTransformer(config).to(config.device)
# model = DecoderOnlyTransformer.from_pretrained('ckpts/DecoderOnlyTransformer').to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

model.train()
for epoch in range(epochs):
    for step, data in enumerate(tqdm(dataloader)):
        src_input_ids = data['input_ids'].to(config.device)
        dst_input_ids = data['labels'].to(config.device)

        outputs = model(src_input_ids, dst_input_ids)
        logits, loss = outputs['logits'], outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tqdm.write(f'loss:{loss.item()}, lr:{scheduler.get_last_lr()[0]}')
        writer.add_scalar('loss', loss.item(), (epoch + 1) * step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], (epoch + 1) * step)

    torch.save({'model_state_dict': model.state_dict(),
                'config': model.config},
                'ckpts/DecoderOnlyTransformer.pth')

print(Colors.BLUE + 'Training Finished!' + Colors.RESET)

idx = model.random_generate()
generated_text = [''.join(tokenizer.decode(x)) for x in idx.tolist()]
print(generated_text)