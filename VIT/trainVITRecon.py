import torch
import torchvision.transforms as T
from transformers import TrainingArguments, Trainer
from config import SimpleVITConfig
from utils import IdenticalDataset, IdenticalCollator
from models import VITForReconstruction

torch.manual_seed(3407)

transform = T.Compose([T.ToTensor(), T.Resize((336, 336)), T.Normalize([0.5], [0.5])])
    
train_dataset = IdenticalDataset('/root/LLM4Beginner/data/VG_100K_2', transform)

config = SimpleVITConfig(hidden_size=256,
                          num_attention_heads=8,
                          n_layers=6,
                          intermediate_size=1024,
                          dropout=0.1,
                          patch_size=14,
                          image_size=336,
                          in_channels=3,
                          flash_attn=True)

model = VITForReconstruction(config).to(config.device)

args = TrainingArguments(output_dir='ckpts/VIT',
                         logging_dir='runs',
                         num_train_epochs=1,
                         per_device_train_batch_size=24,
                         gradient_accumulation_steps=1,
                         save_total_limit=3,
                         logging_steps=10,
                         report_to='tensorboard',
                         bf16=True,
                         learning_rate=1e-3,
                         weight_decay=1e-3,
                         lr_scheduler_type='cosine',
                         dataloader_num_workers=8)

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=train_dataset,
                  data_collator=IdenticalCollator())
trainer.train()
trainer.save_model('ckpts/VIT')
