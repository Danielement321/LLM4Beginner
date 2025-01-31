import torch
from torch.utils.data import Dataset
import time
from glob import glob
from tqdm import tqdm
from PIL import Image
from transformers import TextStreamer
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *

class VLMDataset(Dataset):
    def __init__(self, tokenizer, processor, data_path, config, max_num = None, validate = True, max_validation_workers = 16):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_image_tokens = config.vision_token_num
        self.data = self._load_data(data_path, max_num)
        self.num = int(max_num) if max_num is not None else len(self.data)
        if validate:
            self._validate_data(max_validation_workers)
        else:
            print(Colors.YELLOW + "Data has not been validated, this may cause errors in training!" + Colors.RESET)
        if self.num > len(self.data):
            raise ValueError('num should be less than total data numbers')
        print(f'Total data lines:{self.num:,}')
    
    def _load_data(self, data_folder, max_num):
        lines = []
        print("Reading Data...")
        for file in tqdm(glob(data_folder)):
            with open(file, 'r') as f:
                data = eval(f.read())
                assert type(data) == list
                lines += data
                if max_num is not None and len(lines) == max_num:
                    return lines
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
        image_path = sample['image']
        try:
            Image.open(image_path).close()
            return True
        except Exception as e:
            print(Colors.YELLOW + f"Error in file {image_path}: {e}" + Colors.RESET)
            return False

    def _convert_keys(self, conversations): # Convert the keys of LLaVA dataset
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content = conv["value"]
            messages.append({"role": role, "content": content})
        return messages
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        sample = self.data[index]
        conversations = sample['conversations']
        image_path = sample['image']
        messages = self._convert_keys(conversations)
        formatted_messages = self.tokenizer.apply_chat_template(messages, tokenize = False).replace('<image>', '<|image|>' * self.num_image_tokens)
        # print(formatted_messages)
        tokenized = self.tokenizer(formatted_messages)
        tokenized_input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        input_ids = tokenized_input_ids[:-1]
        labels = tokenized_input_ids[1:]
        pixel_values = self.processor(images = Image.open(image_path).convert('RGB'), return_tensors = 'pt')['pixel_values']
        pixel_values.squeeze_()
        data = {'input_ids': input_ids, 
                'labels': labels, 
                'pixel_values': pixel_values,
                'attention_mask': attention_mask}
        return data
    
class VLMPaddingCollator:
    def __init__(self, tokenizer, max_seq_len = None):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_len = max_seq_len
        
    def __call__(self, features):
        max_seq_len = max(len(x['input_ids']) for x in features)
        input_ids, labels, pixel_values, attention_mask = [], [], [], []
        for data in features:
            input_ids.append(data['input_ids'] + [self.pad_token_id] * (max_seq_len - len(data['input_ids'])))
            labels.append(data['labels'] + [self.pad_token_id] * (max_seq_len - len(data['labels'])))
            attention_mask.append(data['attention_mask'][:max_seq_len] + [0] * (max_seq_len - len(data['attention_mask'])))
            pixel_values.append(data['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor(labels),
                'pixel_values': torch.stack(pixel_values),
                'attention_mask': torch.tensor(attention_mask)}
        
def convert_chat_prompt(prompt, image, tokenizer, processor, config):
    if image is not None:
        if not isinstance(image, list):
            image = [image]
        prompt = "<image>\n" * len(image) + prompt
        prompt = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.replace('<image>', '<|image|>' * config.vision_token_num).strip()}]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt = True, tokenize = False)
        tokenized_prompt = tokenizer(prompt, return_tensors = 'pt')
        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']
        pixel_values = processor(images = image, return_tensors = 'pt')['pixel_values']
        return {'input_ids': input_ids.to(config.device),
                'attention_mask': attention_mask.to(config.device),
                'pixel_values': pixel_values.to(config.device)}
    else:
        print(Colors.YELLOW + "No image is provided. The model will run in text mode." + Colors.RESET)
        prompt = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt = True, tokenize = False)
        tokenized_prompt = tokenizer(prompt, return_tensors = 'pt')
        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']
        return {'input_ids': input_ids.to(config.device),
                'attention_mask': attention_mask.to(config.device)}

def convert_chat_reply(reply, inputs, tokenizer):
    inputs = inputs['input_ids'][0]
    skip_length = inputs.size(0)
    reply = reply[0][skip_length:]
    generated_text = tokenizer.decode(reply, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text

class TTFTTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True):
        super().__init__(tokenizer, skip_prompt)
        self.start_time = None
        self.first_token_time = None
        self.conuter = 0

    def put(self, token_id):
        self.conuter += 1
        if self.start_time is None:
            self.start_time = time.time()
        if self.first_token_time is None and self.conuter == 2:
            self.first_token_time = time.time()
            ttft = self.first_token_time - self.start_time
            print(Colors.GREEN + f"Time to First Token (TTFT): {ttft:.4f} seconds" + Colors.RESET)
        super().put(token_id)