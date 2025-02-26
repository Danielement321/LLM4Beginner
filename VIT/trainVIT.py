import torch
import torchvision
import torchvision.transforms as T
from transformers import TrainingArguments, Trainer
from config import SimpleVITConfig
from utils import ImageClassificationCollator, compute_metrics
from models import VITForClassification

torch.manual_seed(3407)

transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
train_dataset = torchvision.datasets.CIFAR10('data/CIFAR10', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transform, download=True)

config = SimpleVITConfig(hidden_size=256,
                          num_attention_heads=8,
                          n_layers=3,
                          intermediate_size=8,
                          dropout=0.1,
                          patch_size=4,
                          image_size=32,
                          in_channels=3,
                          flash_attn=True)

model = VITForClassification(config, num_classes=10).to(config.device)

args = TrainingArguments(output_dir='ckpts/VIT',
                         logging_dir='runs',
                         num_train_epochs=5,
                         evaluation_strategy="epoch",
                         per_device_train_batch_size=512,
                         per_device_eval_batch_size=256,
                         gradient_accumulation_steps=1,
                         logging_steps=1,
                         report_to='tensorboard',
                         bf16=True,
                         learning_rate=1e-3,
                         weight_decay=1e-3,
                         lr_scheduler_type='cosine',
                         dataloader_num_workers=8)

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  data_collator=ImageClassificationCollator(),
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model('ckpts/VIT')
