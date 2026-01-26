"""
LayerSkip Model

Implements LayerSkip for adaptive layer skipping:
- Early exit mechanism
- Confidence-based layer skipping
- Self-speculative decoding

Variants from Table 5:
- LayerSkip-Llama-3.1-8B: LayerSkip for Llama-3.1-8B

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbeddingL1
# Operators used directly
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants from Table 5
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-8B": {
        "github_repo": "https://github.com/facebookresearch/LayerSkip",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 128256,
        "exit_threshold": 0.9,
    },
}


# ============================================================================
# Wrapper classes for level1 operators that need adaptation
# ============================================================================

class RMSNorm(nn.Module):
    """RMS normalization with learnable weight, using level1 operator."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._rms_norm = RMSNormL1(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding using level1 operator (handles shape transpose)."""
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self._rope = RotaryEmbeddingL1(head_dim, max_seq_len, base)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        q_rotated, k_rotated = self._rope(q_reshaped, k_reshaped)
        return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class EarlyExitClassifier(nn.Module):
    """Classifier for early exit decisions."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden)
        # Return confidence score for exiting
        return self.sigmoid(self.linear(x)).squeeze(-1)


class LayerSkipAttention(nn.Module):
    """Attention using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim)
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LayerSkipMLP(nn.Module):
    """MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class LayerSkipBlock(nn.Module):
    """LayerSkip decoder block with early exit support using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, intermediate_size: int, has_exit: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LayerSkipAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LayerSkipMLP(hidden_size, intermediate_size)
        
        self.has_exit = has_exit
        if has_exit:
            self.exit_classifier = EarlyExitClassifier(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        confidence = None
        if self.has_exit:
            confidence = self.exit_classifier(x)

        return x, confidence


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    LayerSkip model with adaptive layer skipping for efficient inference.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding from level1/embeddings/1_RotaryEmbedding
    - Swish/SiLU from level1/activations/7_Swish
    - Sigmoid from level1/activations/3_Sigmoid
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: Llama-3.1-8B (from Table 5)
    """
    
    VARIANTS = VARIANTS
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 14336,
        num_layers: int = 32,
        vocab_size: int = 128256,
        exit_threshold: float = 0.9,
        exit_layers: Optional[List[int]] = None,
        **kwargs
    ):
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                vocab_size=vocab_size,
            )
        
        super().__init__()
        
        self.exit_threshold = exit_threshold
        
        # Default exit layers at 1/4, 1/2, 3/4 of total layers
        if exit_layers is None:
            exit_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
        self.exit_layers = set(exit_layers)
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Layers with optional early exits
        self.layers = nn.ModuleList([
            LayerSkipBlock(
                hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
                has_exit=(i in self.exit_layers)
            )
            for i in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_early_exit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        x = self.embed_tokens(input_ids)
        
        exit_layer = None
        for i, layer in enumerate(self.layers):
            x, confidence = layer(x)
            
            if use_early_exit and confidence is not None:
                # Check if we can exit early
                mean_confidence = confidence.mean()
                if mean_confidence > self.exit_threshold:
                    exit_layer = i
                    break
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, exit_layer


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
sequence_length = 128
hidden_size = 4096
vocab_size = 128256


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [input_ids]


def get_init_inputs():
    return [{
        'hidden_size': hidden_size,
        'num_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'intermediate_size': 14336,
        'num_layers': 8,
        'vocab_size': vocab_size,
        'exit_threshold': 0.9,
    }]
