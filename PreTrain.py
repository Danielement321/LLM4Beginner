import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
from config import SimpleDecoderOnlyTransformerConfig
from data_utils import PreTrainDataset
from utils import *
from models import DecoderOnlyTransformer

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '<e>'})
config = SimpleDecoderOnlyTransformerConfig(vocab_size=tokenizer.vocab_size + 2, flash_attn=True, max_seq_len=256, hidden_size=512, intermediate_size=2048, num_attention_heads=8)

dataset = PreTrainDataset(tokenizer, './data/*.json', config=config, max_num=1200000)
data_collator = DefaultDataCollator()

config_check(config)
model = DecoderOnlyTransformer(config).to(config.device)

args = TrainingArguments(output_dir='ckpts/PreTrain',
                         logging_dir='runs',
                         num_train_epochs=1, 
                         do_train=True, 
                         per_device_train_batch_size=32,
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

print(model.random_generate())