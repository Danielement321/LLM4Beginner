import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Otherwise there will be warnings, I don't know why.
from utils import set_seed
set_seed(3407)
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoProcessor
from config import SimpleVLMConfig
from data_utils import VLMPTDataset, VLMPaddingCollator
from models import SimpleVLMForConditionalGeneration

config = SimpleVLMConfig()
tokenizer = AutoTokenizer.from_pretrained(config.llm_path)
processor = AutoProcessor.from_pretrained(config.vision_tower_path)
tokenizer.add_tokens(['<|image|>'])
print(f'image_pad_token_id:{tokenizer.encode('<|image|>')}')

dataset = VLMPTDataset(tokenizer, processor, 'data/VLMData/PreTrainData/SimpleChat.json', config)
data_collator = VLMPaddingCollator(tokenizer)

model = SimpleVLMForConditionalGeneration(config)
model.freeze_llm()
model.freeze_vision_tower()

args = TrainingArguments(
    output_dir='ckpts/VLMPreTrain',
    logging_dir='runs',
    num_train_epochs=1, 
    do_train=True, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=1,
    report_to='tensorboard',
    save_steps=5000,
    save_total_limit=5,
    bf16=True,
    learning_rate=2e-3,
    weight_decay=1e-3,
    lr_scheduler_type='cosine',
    dataloader_num_workers=8,
    ddp_find_unused_parameters=False,
    )

trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model('ckpts/VLMPreTrain')
processor.save_pretrained('ckpts/VLMPreTrain')
tokenizer.save_pretrained('ckpts/VLMPreTrain')
print('Finished!')