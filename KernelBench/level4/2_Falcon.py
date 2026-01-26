"""
Falcon Dense Decoder Model

A decoder-only transformer implementing Falcon architecture:
- Multi-Query Attention (MQA) or Grouped-Query Attention (GQA)
- ALiBi (Attention with Linear Biases) position encoding
- LayerNorm normalization
- GELU activation in MLP

Variants from Table 5:
- Falcon-7B: hidden=4544, heads=71, kv_heads=1 (MQA), layers=32
- Falcon-40B: hidden=8192, heads=128, kv_heads=8 (GQA), layers=60

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.attention._6_ALiBi import Model as ALiBi
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "7B": "tiiuae/falcon-7b",
    "40B": "tiiuae/falcon-40b",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class ALiBiPositionBias(nn.Module):
    """ALiBi (Attention with Linear Biases) position encoding."""
    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        
        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", torch.tensor(slopes).view(1, num_heads, 1, 1))
        
        # Pre-compute position differences
        positions = torch.arange(max_seq_len)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        self.register_buffer("rel_pos", rel_pos.unsqueeze(0).unsqueeze(0))
    
    @staticmethod
    def _get_slopes(n_heads: int):
        """Get ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
            return slopes_a + slopes_b

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get ALiBi bias for given sequence length."""
        alibi = self.slopes * self.rel_pos[:, :, :seq_len, :seq_len]
        return alibi


class FalconAttention(nn.Module):
    """Falcon attention with MQA/GQA and ALiBi using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        use_alibi: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        # Falcon uses fused QKV projection
        self.query_key_value = nn.Linear(
            hidden_size, 
            (num_heads + 2 * num_kv_heads) * head_dim, 
            bias=False
        )
        self.dense = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.use_alibi = use_alibi
        if use_alibi:
            self.alibi = ALiBiPositionBias(num_heads, max_seq_len)
        
        # Use level1 MatMul operator
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection
        qkv = self.query_key_value(x)
        
        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for MQA/GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention scores using level1 matmul
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add ALiBi bias
        if self.use_alibi:
            alibi_bias = self.alibi(seq_len)
            attn_weights = attn_weights + alibi_bias

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.dense(attn_output)


class FalconMLP(nn.Module):
    """Falcon MLP with GELU activation using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Use level1 GELU operator
        self.gelu = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.gelu(self.dense_h_to_4h(x)))


class FalconDecoderLayer(nn.Module):
    """Single Falcon decoder layer using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
        use_alibi: bool = True,
    ):
        super().__init__()
        # Use level1 LayerNorm
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attention = FalconAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, max_seq_len, use_alibi
        )
        self.mlp = FalconMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Parallel attention and MLP (Falcon-style)
        residual = x
        x = self.input_layernorm(x)
        
        attn_output = self.self_attention(x, attention_mask)
        mlp_output = self.mlp(x)
        
        x = residual + attn_output + mlp_output
        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Falcon decoder-only transformer with ALiBi.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    
    For HuggingFace integration (weight loading, validation), use FalconAdapter
    from hf_adapters.py.
    """
    
    def __init__(
        self,
        vocab_size: int = 65024,
        hidden_size: int = 4544,
        num_layers: int = 32,
        num_heads: int = 71,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        intermediate_size: int = 18176,
        max_seq_len: int = 2048,
        use_alibi: bool = True,
        **kwargs  # Accept and ignore extra kwargs for flexibility
    ):
        super().__init__()
        
        # Store config values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.use_alibi = use_alibi
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        self.h = nn.ModuleList([
            FalconDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                use_alibi=use_alibi,
            )
            for _ in range(num_layers)
        ])
        
        # Use level1 LayerNorm
        self.ln_f = LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.word_embeddings(input_ids)

        for layer in self.h:
            x = layer(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 2
sequence_length = 512
vocab_size = 65024
hidden_size = 4544
num_layers = 8
num_heads = 71
num_kv_heads = 1
head_dim = 64
intermediate_size = 18176


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'intermediate_size': intermediate_size,
        'alibi': True,
    }]
