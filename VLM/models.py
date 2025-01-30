import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel, GenerationMixin, AutoProcessor, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from config import SimpleVLMConfig, vlm_generation_config
from utils import Colors
from einops import rearrange

class Projector(nn.Module):
    def __init__(self, config: SimpleVLMConfig):
        super().__init__()
        self.config = config
        self.ffn = nn.Sequential(
            nn.Linear(config.vision_tower_hidden_size, config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size)
        )
    
    def forward(self, x):
        return self.ffn(x)

class SimpleVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = SimpleVLMConfig
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    
    def __init__(self, config: SimpleVLMConfig):
        super().__init__(config)
        self.config = config
        self.generation_config = vlm_generation_config
        self.vision_tower = AutoModel.from_pretrained(config.vision_tower_path, _attn_implementation = config._attn_implementation, torch_dtype = torch.bfloat16)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path, _attn_implementation = config._attn_implementation, torch_dtype = torch.bfloat16)
        self.projector = Projector(config)
        print("Model Parameters:", f'{sum([m.numel() for m in self.parameters()]):,}')

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        print(Colors.BLUE + 'The LLM has been frozen!' + Colors.RESET)
        
    def freeze_vision_tower(self):
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        print(Colors.BLUE + 'The Vision Tower has been frozen!' + Colors.RESET)
    
    def get_image_features(self, pixel_values, vision_feature_layer):
        image_outputs = self.vision_tower.vision_model(pixel_values, output_hidden_states = True)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer][:, 1:] # NOTE Here to select image tokens
        image_features = self.projector(selected_image_feature)
        return image_features
    
    def forward(self, input_ids, pixel_values = None, labels = None, attention_mask = None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, self.config.vision_feature_select_layer)
            inputs_embeds = self._merge_input_ids_with_image_features(image_embeds, inputs_embeds, input_ids)
        llm_outputs = self.llm(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
        logits = llm_outputs.logits

        if labels is not None:
            shift_labels = labels[attention_mask != 0]
            shift_logits = logits[attention_mask != 0]
            loss = F.cross_entropy(shift_logits, shift_labels)
            return CausalLMOutput(logits=logits, loss=loss)
        else:
            return CausalLMOutput(logits=logits)
        
    def _merge_input_ids_with_image_features(self, image_embeds, inputs_embeds, input_ids):
        image_embeds = image_embeds.to(inputs_embeds.dtype)
        batch_idx, seq_idx = torch.where(input_ids == self.config.image_pad_token_id)
        inputs_embeds[batch_idx, seq_idx, :] = rearrange(image_embeds, 'b l d -> (b l) d')
        return inputs_embeds

        
def load_model(model_path: str):
    model = SimpleVLMForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model.config.vision_tower_path)
    return tokenizer, processor, model.to(model.config.device)