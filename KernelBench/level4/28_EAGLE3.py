"""
EAGLE-3 Speculative Decoding Model

Implements EAGLE speculative decoding architecture:
- Lightweight draft head for fast token prediction
- Tree attention for parallel verification
- Feature-level speculation

Reference: https://github.com/SafeAILab/EAGLE

Variants from Table 5:
- EAGLE-Llama-3.1-8B: Draft head for Llama-3.1-8B
- EAGLE-Llama-3.1-70B: Draft head for Llama-3.1-70B

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
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from config_loader
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Llama-3.1-8B": "EAGLE3-Llama3-8B",
    "Llama-3.1-70B": "EAGLE3-Llama3-70B",
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
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        q_reshaped = q.transpose(1, 2)
        k_reshaped = k.transpose(1, 2)
        q_rotated, k_rotated = self._rope(q_reshaped, k_reshaped)
        return q_rotated.transpose(1, 2), k_rotated.transpose(1, 2)


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class EAGLEAttention(nn.Module):
    """EAGLE draft head attention using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
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

    def forward(self, x: torch.Tensor, tree_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale

        # Apply tree attention mask for speculative decoding
        if tree_mask is not None:
            attn_weights = attn_weights.masked_fill(~tree_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class EAGLEMLP(nn.Module):
    """EAGLE MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class EAGLEDraftLayer(nn.Module):
    """EAGLE draft layer using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = EAGLEAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = EAGLEMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, tree_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, tree_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class EAGLEDraftHead(nn.Module):
    """EAGLE-style draft head for feature-level speculation using level1 operators."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        # Feature fusion (combines token embedding and hidden state)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        
        # Draft transformer layers
        self.layers = nn.ModuleList([
            EAGLEDraftLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        tree_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fuse hidden states with token embeddings
        x = torch.cat([hidden_states, token_embeddings], dim=-1)
        x = self.fc(x)
        
        # Pass through draft layers
        for layer in self.layers:
            x = layer(x, tree_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    EAGLE speculative decoding model.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding from level1/embeddings/1_RotaryEmbedding
    - Swish/SiLU from level1/activations/7_Swish
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: Llama-3.1-8B, Llama-3.1-70B (configs loaded from config_loader)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "Llama-3.1-8B", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from config_loader."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        hf_config = load_hf_config(VARIANTS[variant])
        hf_config.update(kwargs)
        return cls(operator_level=operator_level, **hf_config)
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        operator_level: Optional[OperatorLevel] = None,
        **kwargs
    ):
        hidden_size = kwargs.get('hidden_size', kwargs.get('draft_hidden', 4096))
        num_heads = kwargs.get('num_heads', 32)
        num_kv_heads = kwargs.get('num_kv_heads', 8)
        head_dim = kwargs.get('head_dim', 128)
        intermediate_size = kwargs.get('intermediate_size', 14336)
        vocab_size = kwargs.get('vocab_size', 128256)
        draft_layers = kwargs.get('draft_layers', 1)
        max_draft_tokens = kwargs.get('max_draft_tokens', kwargs.get('tree_max_width', 60))
        
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
            )
        
        super().__init__()
        
        self.max_draft_tokens = max_draft_tokens
        
        # Token embedding (shared with base model in practice)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Draft head
        self.draft_head = EAGLEDraftHead(
            hidden_size, num_heads, num_kv_heads, head_dim,
            intermediate_size, vocab_size, draft_layers
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        tree_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Base model hidden states (batch, seq, hidden)
            input_ids: Token ids for embedding lookup (batch, seq)
            tree_mask: Optional tree attention mask for speculation
            
        Returns:
            Draft logits (batch, seq, vocab)
        """
        token_embeddings = self.embed_tokens(input_ids)
        return self.draft_head(hidden_states, token_embeddings, tree_mask)
