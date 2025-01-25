import torch
import torchvision
import torchvision.transforms as T
from config import *
from utils import vit_config_check
from models import VITForClassification
from tqdm import tqdm

torch.manual_seed(3407)

epochs = 10
train_batch_size = 512
eval_batch_size = 1024

transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
train_dataset = torchvision.datasets.CIFAR10('data/CIFAR10', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

VIT_CONFIG['image_size'] = 32
VIT_CONFIG['patch_size'] = 4
CONFIG['num_heads'] = 8
CONFIG['encoder_depth'] = 3
CONFIG['d_model'] = 256
CONFIG['ffn_dim'] = 512

model = VITForClassification(VIT_CONFIG, num_classes=10).to(VIT_CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters())

for epoch in range(epochs):
    model.train()
    print(f'Training [{epoch + 1}/{epochs}]...')
    for x, y in tqdm(train_loader):
        x = x.to(VIT_CONFIG['device'])
        y = y.to(VIT_CONFIG['device'])
        pred, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct, total = 0, 0
    model.eval()
    print('Evaluating...')
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(VIT_CONFIG['device'])
            y = y.to(VIT_CONFIG['device'])
            pred = model(x)
            ans = torch.argmax(pred, dim=1)
            correct += (ans == y).sum()
            total += y.shape[0]
    print(f'Accuracy after epoch {epoch + 1}: {(correct/total).item()}')

torch.save(model.state_dict(), 'ckpts/VIT.pth')