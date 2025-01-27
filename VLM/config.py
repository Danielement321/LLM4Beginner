import torch
from transformers import PretrainedConfig, AutoConfig


class SimpleVLMConfig(PretrainedConfig):
    model_type = 'simple_vlm'

    def __init__(self, 
                llm_path = 'Qwen/Qwen2.5-0.5B-Instruct',
                vision_tower_path = 'openai/clip-vit-base-patch16',
                image_pad_token = '<|image|>',
                **kwargs):
        self.llm_path = llm_path
        self.llm_config = AutoConfig.from_pretrained(llm_path)
        self.llm_hidden_size = self.llm_config.hidden_size
        self.vision_tower_path = vision_tower_path
        self.vision_tower_config = AutoConfig.from_pretrained(vision_tower_path)
        self.vision_tower_hidden_size = self.vision_tower_config.vision_config.hidden_size
        self.image_pad_token = image_pad_token
        self.image_pad_token_id = None
        self.vocab_size = self.llm_config.vocab_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(**kwargs)
