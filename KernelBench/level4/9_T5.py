"""
T5 Encoder-Decoder Model

Implements T5 (Text-to-Text Transfer Transformer) architecture:
- Encoder-decoder structure
- Relative position bias
- Pre-norm LayerNorm

Variants from Table 5:
- T5-Base: hidden=768, heads=12, layers=12
- T5-Large: hidden=1024, heads=16, layers=24
- T5-3B: hidden=1024, heads=32, layers=24

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
# Operators used directly
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.matmul._1_MatMul import Model as MatMul
from ..level1.embeddings._4_RelativePositionBias import Model as RelativePositionBiasL1


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Base": "google-t5/t5-base",
    "Large": "google-t5/t5-large",
    "3B": "google-t5/t5-3b",
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class T5LayerNorm(nn.Module):
    """T5-style layer norm with learnable weight, using level1 RMSNorm."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._rms_norm = RMSNormL1(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return self.weight * normalized


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class RelativePositionBias(nn.Module):
    """T5 relative position bias."""
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_len: int, key_len: int, device) -> torch.Tensor:
        query_pos = torch.arange(query_len, device=device)
        key_pos = torch.arange(key_len, device=device)
        relative_position = key_pos.unsqueeze(0) - query_pos.unsqueeze(1)
        
        buckets = self._relative_position_bucket(
            relative_position, self.num_buckets, self.max_distance
        )
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class T5Attention(nn.Module):
    """T5 self-attention with relative position bias using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, has_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, hidden_size, bias=False)
        
        if has_bias:
            self.relative_attention_bias = RelativePositionBias(num_heads)
        else:
            self.relative_attention_bias = None
        
        self.matmul = MatMul()

    def forward(
        self, 
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        is_cross_attention = key_value_states is not None
        kv_states = key_value_states if is_cross_attention else hidden_states
        kv_len = kv_states.shape[1]
        
        q = self.q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_states).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_states).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = self.matmul(q, k.transpose(-2, -1))
        
        if position_bias is None and self.relative_attention_bias is not None:
            position_bias = self.relative_attention_bias(seq_len, kv_len, hidden_states.device)
        
        if position_bias is not None:
            scores = scores + position_bias
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = self.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o(attn_output), position_bias


class T5MLP(nn.Module):
    """T5 dense-relu-dense MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.wi = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.relu(self.wi(x)))


class T5EncoderBlock(nn.Module):
    """T5 encoder block using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, 
                 intermediate_size: int, has_bias: bool = False):
        super().__init__()
        self.layer_norm = T5LayerNorm(hidden_size)
        self.self_attn = T5Attention(hidden_size, num_heads, head_dim, has_bias)
        self.ffn_layer_norm = T5LayerNorm(hidden_size)
        self.ffn = T5MLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None):
        normed = self.layer_norm(x)
        attn_output, position_bias = self.self_attn(normed, position_bias=position_bias, mask=attention_mask)
        x = x + attn_output
        
        normed = self.ffn_layer_norm(x)
        x = x + self.ffn(normed)
        
        return x, position_bias


class T5DecoderBlock(nn.Module):
    """T5 decoder block with cross-attention using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 intermediate_size: int, has_bias: bool = False):
        super().__init__()
        self.self_attn_layer_norm = T5LayerNorm(hidden_size)
        self.self_attn = T5Attention(hidden_size, num_heads, head_dim, has_bias)
        
        self.cross_attn_layer_norm = T5LayerNorm(hidden_size)
        self.cross_attn = T5Attention(hidden_size, num_heads, head_dim, False)
        
        self.ffn_layer_norm = T5LayerNorm(hidden_size)
        self.ffn = T5MLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ):
        normed = self.self_attn_layer_norm(x)
        attn_output, position_bias = self.self_attn(normed, position_bias=position_bias, mask=self_attention_mask)
        x = x + attn_output
        
        normed = self.cross_attn_layer_norm(x)
        cross_output, _ = self.cross_attn(normed, encoder_hidden_states, mask=cross_attention_mask)
        x = x + cross_output
        
        normed = self.ffn_layer_norm(x)
        x = x + self.ffn(normed)
        
        return x, position_bias


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    T5 encoder-decoder transformer.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - ReLU from level1/activations/1_ReLU
    - MatMul from level1/matmul/1_MatMul
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        vocab_size: int = 32128,
        intermediate_size: int = 3072,
        head_dim: int = 64,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        self.shared = nn.Embedding(vocab_size, hidden_size)
        
        self.encoder_blocks = nn.ModuleList([
            T5EncoderBlock(hidden_size, num_heads, head_dim, intermediate_size, has_bias=(i==0))
            for i in range(num_encoder_layers)
        ])
        self.encoder_final_norm = T5LayerNorm(hidden_size)
        
        self.decoder_blocks = nn.ModuleList([
            T5DecoderBlock(hidden_size, num_heads, head_dim, intermediate_size, has_bias=(i==0))
            for i in range(num_decoder_layers)
        ])
        self.decoder_final_norm = T5LayerNorm(hidden_size)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden = self.shared(input_ids)
        position_bias = None
        
        for block in self.encoder_blocks:
            encoder_hidden, position_bias = block(encoder_hidden, position_bias=position_bias)
        
        encoder_hidden = self.encoder_final_norm(encoder_hidden)
        
        decoder_hidden = self.shared(decoder_input_ids)
        seq_len = decoder_input_ids.shape[1]
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        position_bias = None
        for block in self.decoder_blocks:
            decoder_hidden, position_bias = block(
                decoder_hidden, encoder_hidden,
                self_attention_mask=causal_mask,
                position_bias=position_bias,
            )
        
        decoder_hidden = self.decoder_final_norm(decoder_hidden)
        return self.lm_head(decoder_hidden)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
encoder_seq_len = 256
decoder_seq_len = 128
vocab_size = 32128
hidden_size = 768
num_heads = 12
head_dim = 64
intermediate_size = 3072
num_layers = 6


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))
    return [input_ids, decoder_input_ids]


def get_init_inputs():
    return [{
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
        'num_encoder_layers': num_layers,
        'num_decoder_layers': num_layers,
        'vocab_size': vocab_size,
    }]
