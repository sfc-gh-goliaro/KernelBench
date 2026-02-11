"""
Consistency Large Language Model (CLLM)

Implements CLLM for parallel decoding:
- Jacobi trajectory distillation
- Target model for n-token prediction
- Consistency training objective

Variants from Table 5:
- CLLM-Llama-3.1-8B: Consistency model for Llama-3.1-8B

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
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbeddingL1
# Operators used directly
from ..level1.activations._7_Swish import Model as Swish
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants from Table 5
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-8B": {
        "github_repo": "https://github.com/hao-ai-lab/CLLM",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 128256,
        "n_token_prediction": 4,
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

class GlobalProjection(nn.Module):
    """Global projection for n-token prediction."""
    def __init__(self, hidden_size: int, n_predict: int):
        super().__init__()
        self.n_predict = n_predict
        self.projection = nn.Linear(hidden_size, hidden_size * n_predict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden)
        batch, seq, hidden = x.shape
        projected = self.projection(x)
        # Reshape to (batch, seq, n_predict, hidden)
        return projected.view(batch, seq, self.n_predict, hidden)


class CLLMAttention(nn.Module):
    """CLLM attention with causal mask using level1 operators."""
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


class CLLMMLP(nn.Module):
    """MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class CLLMLayer(nn.Module):
    """CLLM decoder layer using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = CLLMAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = CLLMMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Consistency Large Language Model (CLLM) for parallel decoding.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding from level1/embeddings/1_RotaryEmbedding
    - Swish/SiLU from level1/activations/7_Swish
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
        n_token_prediction: int = 4,
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
        
        self.n_token_prediction = n_token_prediction
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Layers
        self.layers = nn.ModuleList([
            CLLMLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        
        # Standard LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # N-token prediction head
        self.global_proj = GlobalProjection(hidden_size, n_token_prediction)
        self.n_token_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(n_token_prediction)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        return_n_predictions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Standard next-token prediction
        logits = self.lm_head(x)
        
        if return_n_predictions:
            # N-token prediction for Jacobi iteration
            projected = self.global_proj(x)
            n_logits = []
            for i, head in enumerate(self.n_token_heads):
                n_logits.append(head(projected[:, :, i, :]))
            n_logits = torch.stack(n_logits, dim=2)  # (batch, seq, n, vocab)
            return logits, n_logits
        
        return logits, None
