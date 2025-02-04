import torch
from torch.utils.data import Dataset
import time
import copy
from constants import *
from glob import glob
from tqdm import tqdm
from PIL import Image
from transformers import TextStreamer
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *

class VLMPTDataset(Dataset):
    def __init__(self, tokenizer, processor, data_path, 
                 config, validate = True, max_validation_workers = 16):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = self._load_data(data_path)
        self.num = len(self.data)
        if validate:
            self._validate_data(max_validation_workers)
        else:
            print(Colors.YELLOW + "Data has not been validated, this may cause errors in training!" + Colors.RESET)
        if self.num > len(self.data):
            raise ValueError('num should be less than total data numbers')
        print(f'{Colors.CYAN}Total data lines:{self.num:,}{Colors.RESET}')
    
    def _load_data(self, data_folder):
        lines = []
        print("Reading Data...")
        for file in tqdm(glob(data_folder)):
            with open(file, 'r') as f:
                data = eval(f.read())
                assert type(data) == list
                lines += data
        if len(lines) == 0:
            raise RuntimeError('Length of training data is 0!')
        return lines
    
    def _validate_data(self, max_validation_workers):
        print("Validating Data...")
        valid_samples = []

        with ThreadPoolExecutor(max_workers=max_validation_workers) as executor:
            futures = {
                executor.submit(self._validate_single_data, sample): sample
                for sample in self.data
            }

            for future in tqdm(as_completed(futures), total=len(self.data)):
                sample = futures[future]
                is_valid = future.result()
                if is_valid:
                    valid_samples.append(sample)

        self.data = valid_samples
        self.num = len(self.data)
    
    def _validate_single_data(self, sample):
        if 'image' in sample.keys():
            image_path = sample['image']
            try:
                Image.open(image_path).close()
                return True
            except Exception as e:
                print(Colors.YELLOW + f"Error in file {image_path}: {e}" + Colors.RESET)
                return False
        else:
            return True

    def _convert_keys(self, conversations): # Convert the keys of LLaVA dataset
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"].replace('<image>', '<|image|>' * VISUAL_TOKEN_NUM)
            messages.append({"role": role, "content": content})
        return messages
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        sample = self.data[index]
        conversations = sample['conversations']
        messages = self._convert_keys(conversations)
        tokenized = self.tokenizer.apply_chat_template(messages, return_dict = True)
        tokenized_input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        loss_mask = copy.deepcopy(attention_mask)
        input_ids = tokenized_input_ids[:-1]
        labels = tokenized_input_ids[1:]
        if 'image' in sample.keys():
            image_path = sample['image']
            pixel_values = self.processor(images = Image.open(image_path).convert('RGB'), return_tensors = 'pt')['pixel_values']
            pixel_values.squeeze_()
        else:
            pixel_values = None
        return {'input_ids': input_ids, 
                'labels': labels, 
                'pixel_values': pixel_values,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask}

class VLMSFTDataset(VLMPTDataset):
    def __init__(self, tokenizer, processor, data_path, config,
                 validate = True, max_validation_workers = 16):
        super().__init__(tokenizer, processor, data_path, config, 
                         validate, max_validation_workers)
    
    def get_assistant_tokens_mask(self, input_ids, attention_mask):
        starts, ends = [], []
        for i in range(len(input_ids) - 1):
            if input_ids[i] == BOS_TOKEN_ID and input_ids[i + 1] == ASSISTANT_TOKEN_ID:
                starts.append(i)
                for j in range(i+1, len(input_ids)):
                    if input_ids[j] == EOS_TOKEN_ID:
                        ends.append(j)
                        break
        assistant_tokens_mask = [0] * len(attention_mask)
        
        if len(starts) != len(ends) or len(starts) * len(ends) == 0: 
            # For invalid data, compute loss for the whole sequence
            print(Colors.YELLOW + "Current SFT data is invalid." + Colors.RESET)
            assistant_tokens_mask = [1] * len(attention_mask)
            return assistant_tokens_mask

        for i, j in zip(starts, ends):
            assistant_tokens_mask[i: j + 1] = [1] * (j - i)
        return assistant_tokens_mask
    
    def __getitem__(self, index):
        sample = self.data[index]
        conversations = sample['conversations']
        messages = self._convert_keys(conversations)
        tokenized = self.tokenizer.apply_chat_template(messages, return_dict = True)
        loss_mask = self.get_assistant_tokens_mask(tokenized['input_ids'], tokenized['attention_mask'])
        tokenized_input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        input_ids = tokenized_input_ids[:-1]
        labels = tokenized_input_ids[1:]
        if 'image' in sample.keys():
            image_path = sample['image']
            pixel_values = self.processor(images = Image.open(image_path).convert('RGB'), return_tensors = 'pt')['pixel_values']
            pixel_values.squeeze_()
        else:
            pixel_values = None
        return {'input_ids': input_ids, 
                'labels': labels, 
                'pixel_values': pixel_values,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask}


class VLMPaddingCollator:
    def __init__(self, tokenizer, max_seq_len = None):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_len = max_seq_len
        
    def __call__(self, features):
        max_seq_len = max(len(x['input_ids']) for x in features)
        input_ids, labels, pixel_values, attention_mask, loss_mask = [], [], [], [], []
        for data in features:
            input_ids.append(data['input_ids'] + [self.pad_token_id] * (max_seq_len - len(data['input_ids'])))
            labels.append(data['labels'] + [self.pad_token_id] * (max_seq_len - len(data['labels'])))
            attention_mask.append(data['attention_mask'][:max_seq_len] + [0] * (max_seq_len - len(data['attention_mask'])))
            loss_mask.append(data['loss_mask'][:max_seq_len] + [0] * (max_seq_len - len(data['loss_mask'])))
            if data['pixel_values'] is not None:
                pixel_values.append(data['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor(labels),
                'pixel_values': torch.stack(pixel_values),
                'attention_mask': torch.tensor(attention_mask),
                'loss_mask': torch.tensor(loss_mask)}
        
def convert_chat_prompt(prompt, image, tokenizer, processor, config):
    if image is not None:
        if not isinstance(image, list):
            image = [image]
        prompt = "<image>\n" * len(image) + prompt
        prompt = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.replace('<image>', '<|image|>' * VISUAL_TOKEN_NUM).strip()}]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt = True, return_tensors = 'pt', return_dict = True)
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask']
        pixel_values = processor(images = image, return_tensors = 'pt')['pixel_values']
        return {'input_ids': input_ids.to(config.device),
                'attention_mask': attention_mask.to(config.device),
                'pixel_values': pixel_values.to(config.device)}
    else:
        print(Colors.YELLOW + "No image is provided. The model will run in text mode." + Colors.RESET)
        prompt = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt = True, return_tensors = 'pt', return_dict = True)
        input_ids = prompt['input_ids']
        attention_mask = prompt['attention_mask']
        return {'input_ids': input_ids.to(config.device),
                'attention_mask': attention_mask.to(config.device)}

def convert_chat_reply(reply, inputs, tokenizer):
    inputs = inputs['input_ids'][0]
    skip_length = inputs.size(0)
    reply = reply[0][skip_length:]
    generated_text = tokenizer.decode(reply, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text

class TextStreamerWithTime(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt)
        self.start_time = None
        self.first_token_time = None
        self.counter = 0

    def put(self, token_id):
        self.counter += 1
        if self.start_time is None:
            self.start_time = time.time()
        if self.first_token_time is None and self.counter == 2:
            self.first_token_time = time.time()
            self.ttft = self.first_token_time - self.start_time
        super().put(token_id)
    
    def end(self):
        super().end()
        print(Colors.BLUE + f"Time to First Token: {self.ttft:.4f} s" + Colors.RESET)
        print(Colors.BLUE + f'Time Per Token: {(time.time() - self.start_time)/self.counter:.4f}s' + Colors.RESET)