import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
from config import SimpleDecoderOnlyTransformerConfig
from data_utils import DatasetForCasualLM
from utils import *
from models import DecoderOnlyTransformer

sample_size = 10000000

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
config = SimpleDecoderOnlyTransformerConfig(vocab_size=tokenizer.vocab_size, flash_attn=False)

dataset = DatasetForCasualLM(tokenizer, num=sample_size, config=config)
data_collator = DefaultDataCollator()

config_check(config)
model = DecoderOnlyTransformer(config).to(config.device)

args = TrainingArguments(output_dir='ckpts/DecoderOnlyTransformer', 
                        num_train_epochs=1, 
                        do_train=True, 
                        per_device_train_batch_size=96,
                        gradient_accumulation_steps=2,
                        # max_steps=15000,
                        logging_steps=1,
                        report_to='tensorboard',
                        save_total_limit=5,
                        bf16=True,
                        learning_rate=8e-4,
                        lr_scheduler_type='cosine',
                        dataloader_num_workers=8,
                        dataloader_pin_memory=True,
                        save_safetensors=False)

trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model('ckpts/DecoderOnlyTransformer')