import torch
import torch.nn as nn
from einops import rearrange


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear3 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.silu = nn.SiLU()
        self.norm = RMSNorm(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_ = x.clone()
        x = self.norm(x)
        gate = self.silu(self.linear1(x))
        x = self.linear2(x) * gate
        x = self.linear3(x)
        x = self.dropout(x)
        return x + x_


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        x = x * self.weight
        return x


# Complex64 can not be saved in safetensor, so this class will not inherit from nn.Module
class RotaryEmbedding():
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % 2 == 0
        head_size = config.hidden_size // config.num_attention_heads
        freqs = 1 / (10000 ** (torch.arange(0, head_size, 2) / head_size))
        t = torch.arange(0, config.max_seq_len).float()
        freqs = torch.outer(t, freqs) # Equivalent to t.unsqueeze(1) @ freqs.unsqueeze(0), but maybe faster
        pos_cis = torch.polar(torch.ones_like(freqs), freqs).to(config.device)
        self.pos_cis = rearrange(pos_cis, f'l d -> 1 l 1 d') # [max_seq_len, hidden_size//2] -> [1 max_seq_len 1 hidden_size//2]
    
    def embed(self, q, k):
        seq_len = q.shape[1]
        q_ = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2).float()) # [b l h k] -> [b l h k//2 2] -> [b l h k//2]
        k_ = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2).float()) # [b l h k] -> [b l h k//2 2] -> [b l h k//2]
        q_out = torch.view_as_real(q_ * self.pos_cis[:, :seq_len, :, :]).flatten(3) # [b l h k//2] -> [b l h k//2 2] -> [b l h k]
        k_out = torch.view_as_real(k_ * self.pos_cis[:, :seq_len, :, :]).flatten(3) # [b l h k//2] -> [b l h k//2 2] -> [b l h k]
        return q_out, k_out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.config = config
        self.d_k = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.softmax = nn.Softmax(dim=-1)
        # self.RoPE = RotaryEmbedding(config)
        self.Q_norm = RMSNorm(config)
        self.K_norm = RMSNorm(config)
        self.V_norm = RMSNorm(config)
        self.Q_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.K_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.V_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_weights = None
    
    def _process_qkv(self, q, k, v):
        q = self.Q_norm(q)
        k = self.K_norm(k)
        v = self.V_norm(v)

        q = self.Q_map(q)
        k = self.K_map(k)
        v = self.V_map(v)

        q = rearrange(q, 'b l (h k) -> b l h k', h = self.num_heads)
        k = rearrange(k, 'b l (h k) -> b l h k', h = self.num_heads)
        v = rearrange(v, 'b l (h k) -> b h l k', h = self.num_heads)
        # q, k = self.RoPE.embed(q, k)
        q = rearrange(q, 'b l h k -> b h l k')
        k = rearrange(k, 'b l h k -> b h l k')
        return q, k, v
    
    def attention(self, q, k, v, mask):
        q, k, v = self._process_qkv(q, k, v)

        if mask is not None:
            score = self.softmax((q @ k.transpose(-1, -2))/self.d_k**0.5 + mask) @ v
        else:
            score = self.softmax((q @ k.transpose(-1, -2))/self.d_k**0.5) @ v
            
        output = rearrange(score, 'b h l k -> b l (h k)')
        return output
    
    def flash_attention(self, q, k, v, mask=None):
        q, k, v = self._process_qkv(q, k, v)

        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.config.dropout)

        output = rearrange(output, 'b h l k -> b l (h k)')
        return output
    
    def attention_with_weights(self, q, k, v, mask):
        q, k, v = self._process_qkv(q, k, v)

        scores = (q @ k.transpose(-1, -2)) / self.d_k**0.5 + mask

        self.attention_weights = self.softmax(scores)
        score = self.attention_weights @ v
            
        output = rearrange(score, 'b h l k -> b l (h k)')
        return output


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)
        self.flash_attn = config.flash_attn

    def forward(self, x, mask):
        x_ = x.clone()
        if self.flash_attn:
            x = self.flash_attention(q=x, k=x, v=x)
        else:
            x = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.fc(x)
        x = self.dropout(x)
        return x + x_


class MultiHeadSelfAttentionWithMap(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x, mask):
        x_ = x.clone()
        x = self.attention_with_weights(q=x, k=x, v=x, mask=mask)
        x = self.fc(x)
        x = self.dropout(x)
        return x + x_


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, src, dst, mask):
        dst_ = dst.clone()
        dst = self.attention(q=dst, k=src, v=src, mask=mask)
        dst = self.fc(dst)
        dst = self.dropout(dst)
        return dst + dst_


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.ffn = FFN(config)
    
    def forward(self, src, mask = None):
        src = self.attention(src, mask)
        src = self.ffn(src)
        return src


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.d_model = config.hidden_size
        # self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layers)])
    
    def forward(self, x, mask = None, embed = True):
        if embed:
            x = self.embed(x)
            x = self.position_encode(x)

        for block in self.encoder_blocks:
            x = block(x, mask)
        return x
    
    @torch.no_grad()
    def position_encode(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len).unsqueeze(1)  # [seq_len, 1]
        div_term = 10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model)  # [d_model/2]
        sin_values = torch.sin(positions / div_term).to(self.device)  # [seq_len, d_model/2]
        cos_values = torch.cos(positions / div_term).to(self.device)  # [seq_len, d_model/2]
        position_embedding = torch.zeros_like(x, requires_grad=False).to(self.device)
        position_embedding[:, :, 0::2] = sin_values.unsqueeze(0)  # [batch_size, seq_len, d_model/2]
        position_embedding[:, :, 1::2] = cos_values.unsqueeze(0)  # [batch_size, seq_len, d_model/2]
        x += position_embedding
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(config)
        self.ffn = FFN(config)
    
    def forward(self, dst, causal_mask = None):
        dst = self.self_attention(dst, causal_mask)
        dst = self.ffn(dst)
        return dst


class DecoderBlockWithCrossAttention(DecoderBlock):
    def __init__(self, config):
        super().__init__()
        self.cross_attention = MultiHeadCrossAttention(config)
    
    def forward(self, src, dst, padding_mask = None, causal_mask = None):
        dst = self.self_attention(dst, causal_mask)
        dst = self.cross_attention(src, dst, padding_mask)
        dst = self.ffn(dst)
        return dst


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.d_model = config.hidden_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
    
    def forward(self, dst, causal_mask = None):
        dst = self.embed(dst)
        # dst = self.position_encode(dst) # Sinusoidal is deprecated since RoPE is applied
        for block in self.decoder_blocks:
            dst = block(dst = dst, causal_mask = causal_mask)
        return dst
    
    @torch.no_grad()
    def position_encode(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len).unsqueeze(1)  # [seq_len, 1]
        div_term = 10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model)  # [d_model/2]
        sin_values = torch.sin(positions / div_term).to(self.device)  # [seq_len, d_model/2]
        cos_values = torch.cos(positions / div_term).to(self.device)  # [seq_len, d_model/2]
        position_embedding = torch.zeros_like(x, requires_grad=False).to(self.device)
        position_embedding[:, :, 0::2] = sin_values.unsqueeze(0)  # [batch_size, seq_len, d_model/2]
        position_embedding[:, :, 1::2] = cos_values.unsqueeze(0)  # [batch_size, seq_len, d_model/2]
        x += position_embedding
        return x

class DecoderWithCrossAttention(Decoder):
    def __init__(self, config):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlockWithCrossAttention(config) for _ in range(config.n)])
    
    def forward(self, src, dst, padding_mask = None, causal_mask = None):
        dst = self.embed(dst)
        dst = self.position_encode(dst)
        for block in self.decoder_blocks:
            dst = block(src, dst, padding_mask, causal_mask)
        return dst
