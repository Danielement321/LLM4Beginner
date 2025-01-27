import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Otherwise there will be warnings, I don't know why.

from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoProcessor
from config import SimpleVLMConfig
from data_utils import VLMDataset, VLMPaddingCollator
from utils import *
from models import SimpleVLMForConditionalGeneration

config = SimpleVLMConfig()
tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
processor = AutoProcessor.from_pretrained(config.vision_tower_path)
tokenizer.add_tokens(['<|image|>'])
config.image_pad_token_id = tokenizer.encode(config.image_pad_token)[0]

dataset = VLMDataset(tokenizer, processor, 'data/VLMData/*.json', config)
data_collator = VLMPaddingCollator(tokenizer)

config_check(config)
model = SimpleVLMForConditionalGeneration(config).to(config.device)
model.freeze_llm()
model.freeze_vision_tower()

args = TrainingArguments(
    output_dir='ckpts/VLMPreTrain',
    logging_dir='runs',
    num_train_epochs=1, 
    do_train=True, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    report_to='tensorboard',
    save_steps=5000,
    save_total_limit=5,
    bf16=True,
    learning_rate=1e-3,
    weight_decay=1e-3,
    lr_scheduler_type='cosine',
    dataloader_num_workers=8,
    )

trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model('ckpts/VLMPreTrain')
print('Finished!')