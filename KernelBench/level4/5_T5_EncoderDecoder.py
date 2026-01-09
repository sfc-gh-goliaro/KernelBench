"""
T5 Encoder-Decoder Transformer

A sequence-to-sequence transformer implementing T5 architecture:
- Encoder-decoder structure with cross-attention
- Relative position bias (not learned embeddings)
- Pre-norm architecture
- Shared embedding for encoder/decoder

Reference: T5-large
- Hidden: 1024, Heads: 16, FFN: 2816, Layers: 24 (each encoder/decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class T5LayerNorm(nn.Module):
    """T5-style Layer Normalization (no bias, no mean subtraction)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class T5RelativePositionBias(nn.Module):
    """Relative position bias for T5 attention."""
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        relative_buckets = 0
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).long() * num_buckets
            relative_position = relative_position.abs()
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position

        relative_buckets = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_buckets)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values


class T5Attention(nn.Module):
    """T5 self-attention with relative position bias."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        has_relative_bias: bool = True,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_decoder = is_decoder

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.relative_bias = T5RelativePositionBias(
                num_heads, bidirectional=not is_decoder
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))

        # Add position bias
        if position_bias is None and self.has_relative_bias:
            position_bias = self.relative_bias(seq_len, seq_len, hidden_states.device)
        if position_bias is not None:
            scores = scores + position_bias

        # Causal mask for decoder
        if self.is_decoder:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output), position_bias


class T5CrossAttention(nn.Module):
    """T5 cross-attention for encoder-decoder."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = hidden_states.shape
        src_len = encoder_hidden_states.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.hidden_size)
        return self.o_proj(attn_output)


class T5FFN(nn.Module):
    """T5 Feed-forward network with gated activation."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.wi_0(x), approximate='tanh') * self.wi_1(x)
        hidden = self.dropout(hidden)
        return self.wo(hidden)


class T5EncoderLayer(nn.Module):
    """Single T5 encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, has_relative_bias: bool = True):
        super().__init__()
        self.self_attn = T5Attention(hidden_size, num_heads, has_relative_bias, is_decoder=False)
        self.self_attn_norm = T5LayerNorm(hidden_size)
        self.ffn = T5FFN(hidden_size, intermediate_size)
        self.ffn_norm = T5LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        normed = self.self_attn_norm(x)
        attn_out, position_bias = self.self_attn(normed, attention_mask, position_bias)
        x = x + attn_out

        # FFN
        normed = self.ffn_norm(x)
        x = x + self.ffn(normed)

        return x, position_bias


class T5DecoderLayer(nn.Module):
    """Single T5 decoder layer with cross-attention."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, has_relative_bias: bool = True):
        super().__init__()
        self.self_attn = T5Attention(hidden_size, num_heads, has_relative_bias, is_decoder=True)
        self.self_attn_norm = T5LayerNorm(hidden_size)

        self.cross_attn = T5CrossAttention(hidden_size, num_heads)
        self.cross_attn_norm = T5LayerNorm(hidden_size)

        self.ffn = T5FFN(hidden_size, intermediate_size)
        self.ffn_norm = T5LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        normed = self.self_attn_norm(x)
        attn_out, position_bias = self.self_attn(normed, attention_mask, position_bias)
        x = x + attn_out

        # Cross-attention
        normed = self.cross_attn_norm(x)
        cross_out = self.cross_attn(normed, encoder_hidden_states, encoder_attention_mask)
        x = x + cross_out

        # FFN
        normed = self.ffn_norm(x)
        x = x + self.ffn(normed)

        return x, position_bias


class Model(nn.Module):
    """T5 Encoder-Decoder model."""
    def __init__(
        self,
        vocab_size: int = 32128,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 2816,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Shared embeddings
        self.shared_embedding = nn.Embedding(vocab_size, hidden_size)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(T5EncoderLayer(
                hidden_size, num_heads, intermediate_size,
                has_relative_bias=(i == 0)  # Only first layer has bias
            ))
        self.encoder_norm = T5LayerNorm(hidden_size)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(T5DecoderLayer(
                hidden_size, num_heads, intermediate_size,
                has_relative_bias=(i == 0)
            ))
        self.decoder_norm = T5LayerNorm(hidden_size)

        # LM head (tied with embedding)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder
        encoder_hidden = self.shared_embedding(input_ids)
        position_bias = None
        for layer in self.encoder_layers:
            encoder_hidden, position_bias = layer(encoder_hidden, attention_mask, position_bias)
        encoder_hidden = self.encoder_norm(encoder_hidden)

        # Decoder
        decoder_hidden = self.shared_embedding(decoder_input_ids)
        position_bias = None
        for layer in self.decoder_layers:
            decoder_hidden, position_bias = layer(
                decoder_hidden, encoder_hidden, None, attention_mask, position_bias
            )
        decoder_hidden = self.decoder_norm(decoder_hidden)

        # LM head
        return self.lm_head(decoder_hidden)


# Configuration (reduced for benchmarking)
batch_size = 4
src_seq_length = 512
tgt_seq_length = 128
vocab_size = 32128
hidden_size = 768
num_layers = 6
num_heads = 12
intermediate_size = 2048


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, src_seq_length))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_length))
    return [input_ids, decoder_input_ids]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'intermediate_size': intermediate_size,
    }]

