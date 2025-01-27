import torch
from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from config import SimpleVLMConfig

class VLMDataset(Dataset):
    def __init__(self, tokenizer, processor, data_path, config: SimpleVLMConfig, max_num = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_image_tokens = 197 # TODO This should be changed accordingly
        self.data = self._load_data(data_path, max_num)
        self.num = int(max_num) if max_num is not None else len(self.data)
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
    
    def __len__(self):
        return self.num
    
    def __getitem__(self, index):
        sample = self.data[index]
        messages = sample['messages']
        image_path = sample['images']
        formatted_messages = self.tokenizer.apply_chat_template(messages, tokenize = False).replace('<image>', self.config.image_pad_token * self.num_image_tokens)
        # print(formatted_messages)
        tokenized = self.tokenizer(formatted_messages)
        tokenized_input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        input_ids = tokenized_input_ids[:-1]
        labels = tokenized_input_ids[1:]
        # images = [Image.open(i) for i in image_path] # TODO support multiple images
        # pixel_values = self.processor(images = images, return_tensors = 'pt')['pixel_values']
        pixel_values = self.processor(images = Image.open(image_path[0]).convert('RGB'), return_tensors = 'pt')['pixel_values']
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
                'pixel_values': torch.stack(pixel_values, dim=0),
                'attention_mask': None} # TODO 