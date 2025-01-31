import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Otherwise there will be warnings, I don't know why.
from utils import set_seed
set_seed(3407)
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoProcessor
from data_utils import VLMSFTDataset, VLMPaddingCollator
from models import SimpleVLMForConditionalGeneration

pre_trained_path = 'ckpts/VLMPreTrain'
tokenizer = AutoTokenizer.from_pretrained(pre_trained_path)
processor = AutoProcessor.from_pretrained(pre_trained_path)

model = SimpleVLMForConditionalGeneration.from_pretrained(pre_trained_path)
model.freeze_vision_tower()

dataset = VLMSFTDataset(tokenizer, processor, 'data/VLMData/PreTrainData/SimpleChat.json', model.config)
data_collator = VLMPaddingCollator(tokenizer)

args = TrainingArguments(
    output_dir='ckpts/VLMFinetune',
    logging_dir='runs',
    num_train_epochs=3, 
    do_train=True, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=1,
    report_to='tensorboard',
    save_strategy='epoch',
    save_total_limit=5,
    bf16=True,
    learning_rate=1e-4,
    weight_decay=1e-3,
    lr_scheduler_type='cosine',
    dataloader_num_workers=8,
    ddp_find_unused_parameters=False,
    )

trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=data_collator)
trainer.train()
trainer.save_model('ckpts/VLMFinetune')
processor.save_pretrained('ckpts/VLMFinetune')
tokenizer.save_pretrained('ckpts/VLMFinetune')
print('Finished!')