import torch
from transformers import PretrainedConfig, AutoConfig, GenerationConfig


class SimpleVLMConfig(PretrainedConfig):
    model_type = 'simple_vlm'

    def __init__(self, 
                llm_path = 'Qwen/Qwen2.5-0.5B-Instruct',
                vision_tower_path = 'openai/clip-vit-base-patch16',
                image_pad_token_id = 151665,
                vision_token_num = 196,
                vision_feature_select_layer = -1,
                flash_attention = True,
                **kwargs):
        super().__init__(**kwargs)
        self.llm_path = llm_path
        self.llm_config = AutoConfig.from_pretrained(llm_path)
        self.llm_hidden_size = self.llm_config.hidden_size
        self.vision_tower_path = vision_tower_path
        self.vision_tower_config = AutoConfig.from_pretrained(vision_tower_path)
        self.vision_tower_hidden_size = self.vision_tower_config.vision_config.hidden_size
        self.image_pad_token_id = image_pad_token_id
        self.vocab_size = self.llm_config.vocab_size
        self.vision_token_num = vision_token_num
        self.vision_feature_select_layer = vision_feature_select_layer
        self._attn_implementation = 'flash_attention_2' if flash_attention else 'sdpa'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The special tokens are identical to Qwen
vlm_generation_config = GenerationConfig(
    bos_token_id = 151643,
    pad_token_id = 151643,
    eos_token_id = [151645, 151643],
    )