import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from config import SimpleVLMConfig
from modules import *
from utils import Colors
from einops import rearrange

class SimpleVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = SimpleVLMConfig
    
    def __init__(self, config: SimpleVLMConfig):
        super().__init__(config)
        self.config = config
        self.vision_tower = AutoModel.from_pretrained(config.vision_tower_path)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path)
        self.projector = Projector(config)
        
    def forward(self, input_ids, pixel_values = None, labels = None, attention_mask = None):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_features = self.vision_tower.vision_model(pixel_values).last_hidden_state
        image_embeds = self.projector(image_features).to(text_embeds.dtype)
        inputs_embeds = self._merge_input_ids_with_image_features(image_embeds, text_embeds, input_ids)
        llm_outputs = self.llm(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
        logits = llm_outputs.logits

        if labels is not None:
            loss = F.cross_entropy(rearrange(logits, 'b l d -> (b l) d'), rearrange(labels, 'b l -> (b l)'))
            return CausalLMOutput(logits=logits, loss=loss)
        else:
            return CausalLMOutput(logits=logits)
        
    def _merge_input_ids_with_image_features(self, image_embeds, inputs_embeds, input_ids):
        batch_idx, seq_idx = torch.where(input_ids == self.config.image_pad_token_id)
        inputs_embeds[batch_idx, seq_idx, :] = rearrange(image_embeds, 'b l d -> (b l) d')
        return inputs_embeds
    
    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        print(Colors.BLUE + 'The LLM has been frozen!' + Colors.RESET)
        
    def freeze_vision_tower(self):
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        print(Colors.BLUE + 'The Vision Tower has been frozen!' + Colors.RESET)