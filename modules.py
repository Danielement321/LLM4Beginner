import torch
import torch.nn as nn
from einops import rearrange


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=config['d_model'], out_features=config['ffn_dim']),
            nn.ReLU(),
            nn.Linear(in_features=config['ffn_dim'], out_features=config['d_model']),
            nn.Dropout(p=config['dropout']),
        )
        self.norm = nn.LayerNorm(config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x_ = x.clone()
        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return x + x_


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['d_model'] % config['num_heads'] == 0
        self.d_k = config['d_model'] // config['num_heads']
        self.num_heads = config['num_heads']
        self.softmax = nn.Softmax(dim=-1)
        self.Q_norm = nn.LayerNorm(config['d_model'])
        self.K_norm = nn.LayerNorm(config['d_model'])
        self.V_norm = nn.LayerNorm(config['d_model'])
        self.Q_map = nn.Linear(config['d_model'], config['d_model'])
        self.K_map = nn.Linear(config['d_model'], config['d_model'])
        self.V_map = nn.Linear(config['d_model'], config['d_model'])
        self.fc = nn.Linear(config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def attention(self, q, k, v, mask):
        q = self.Q_norm(q)
        k = self.K_norm(k)
        v = self.V_norm(v)

        q = self.Q_map(q)
        k = self.K_map(k)
        v = self.V_map(v)

        q = rearrange(q, 'b l (h k) -> b h l k', h = self.num_heads)
        k = rearrange(k, 'b l (h k) -> b h k l', h = self.num_heads)
        v = rearrange(v, 'b l (h k) -> b h l k', h = self.num_heads)

        if mask is not None:
            score = self.softmax((q @ k + mask)/self.d_k**0.5) @ v
        else:
            score = self.softmax((q @ k)/self.d_k**0.5) @ v

        score = rearrange(score, 'b h l k -> b l (h k)')
        return score

    def forward(self):
        raise NotImplementedError("This method must be implemented in downstream task!")


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, src, padding_mask):
        src_ = src.clone()
        src = self.attention(q=src, k=src, v=src, mask=padding_mask)
        src = self.fc(src)
        src = self.dropout(src)
        return src + src_


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, src, dst, padding_mask):
        dst_ = dst.clone()
        dst = self.attention(q=dst, k=src, v=src, mask=padding_mask)
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
        self.device = config['device']
        self.d_model = config['d_model']
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config['encoder_depth'])])
    
    def forward(self, x, mask = None):
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
        self.cross_attention = MultiHeadCrossAttention(config)
        self.ffn = FFN(config)
    
    def forward(self, src, dst, padding_mask = None, casual_mask = None):
        dst = self.self_attention(dst, casual_mask)
        if src is not None: # For decoder-only architecture
            dst = self.cross_attention(src, dst, padding_mask)
        dst = self.ffn(dst)
        return dst


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.d_model = config['d_model']
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config['decoder_depth'])])
    
    def forward(self, src, dst, padding_mask = None, casual_mask = None):
        dst = self.embed(dst)
        dst = self.position_encode(dst)
        for block in self.decoder_blocks:
            dst = block(src, dst, padding_mask, casual_mask)
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
