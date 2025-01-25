import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
from config import SimpleDecoderOnlyTransformerConfig
from data_utils import PreTrainDataset
from utils import *
from models import DecoderOnlyTransformer

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
config = SimpleDecoderOnlyTransformerConfig(vocab_size=tokenizer.vocab_size, flash_attn=True)

dataset = PreTrainDataset(tokenizer, './data/*.json', config=config, max_num=10000)
data_collator = DefaultDataCollator()

config_check(config)
model = DecoderOnlyTransformer(config).to(config.device)

args = TrainingArguments(output_dir='ckpts/PreTrain',
                         logging_dir='runs',
                         num_train_epochs=1, 
                         do_train=True, 
                         per_device_train_batch_size=64,
                         gradient_accumulation_steps=2,
                         logging_steps=1,
                         report_to='tensorboard',
                         save_steps=5000,
                         save_total_limit=5,
                         bf16=True,
                         learning_rate=1e-3,
                         weight_decay=1e-3,
                         lr_scheduler_type='cosine',
                         dataloader_num_workers=8)

trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model('ckpts/PreTrain')