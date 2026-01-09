"""
Qwen2.5 Reward Model

A reward model for RLHF based on Qwen2.5 architecture:
- Qwen2.5 decoder backbone (no LM head)
- Last token pooling
- Linear projection to scalar score
- Optional regression head with activation

Reference: Qwen2.5-Math-RM, Skywork-Reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(N)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    """Decoder block."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LastTokenPooling(nn.Module):
    """Extract the last non-padding token representation."""
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            # Use last token
            return hidden_states[:, -1]

        # Find last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths.long()]


class ScoreHead(nn.Module):
    """Score head for reward model."""
    def __init__(self, hidden_size: int, use_activation: bool = False):
        super().__init__()
        if use_activation:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
        else:
            self.head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class Model(nn.Module):
    """Qwen2.5 Reward Model."""
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_layers: int = 28,
        num_heads: int = 28,
        num_kv_heads: int = 4,
        intermediate_size: int = 18944,
        use_activation_in_head: bool = False,
    ):
        super().__init__()

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers (same as Qwen2.5)
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_size)

        # Pooling and score head (replaces LM head)
        self.pooler = LastTokenPooling()
        self.score_head = ScoreHead(hidden_size, use_activation_in_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Create attention mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
        else:
            extended_mask = None

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_mask)

        hidden_states = self.norm(hidden_states)

        # Pool and score
        pooled = self.pooler(hidden_states, attention_mask)
        score = self.score_head(pooled)

        return score


# Configuration (reduced for benchmarking)
batch_size = 4
sequence_length = 1024
vocab_size = 152064
hidden_size = 2048
num_layers = 8
num_heads = 16
num_kv_heads = 4
intermediate_size = 5504


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    # Simulate variable-length sequences
    for i in range(batch_size):
        pad_len = torch.randint(0, 100, (1,)).item()
        if pad_len > 0:
            attention_mask[i, -pad_len:] = 0
    return [input_ids, attention_mask]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'intermediate_size': intermediate_size,
        'use_activation_in_head': False,
    }]

