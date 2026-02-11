"""
Lookahead Decoding Model

Implements Lookahead speculative decoding:
- N-gram based speculation
- Jacobi iteration for parallel verification
- No auxiliary draft model required

Variants from Table 5:
- Lookahead-Llama-3.1-8B: Lookahead for Llama-3.1-8B

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
from ..level1.activations._5_Softmax import Model as Softmax
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants - configs loaded from config_loader
# ============================================================================

VARIANTS: Dict[str, str] = {
    "W4-N5": "lookahead-W4-N5",
    "W7-N7": "lookahead-W7-N7",
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

class NgramPool(nn.Module):
    """N-gram pool for storing and retrieving speculation candidates."""
    def __init__(self, max_size: int = 10000, n: int = 4):
        super().__init__()
        self.max_size = max_size
        self.n = n
        self.ngrams: Dict[Tuple[int, ...], List[int]] = {}

    def add(self, tokens: List[int]) -> None:
        """Add tokens to the n-gram pool."""
        for i in range(len(tokens) - self.n):
            key = tuple(tokens[i:i+self.n])
            continuation = tokens[i+self.n]
            if key not in self.ngrams:
                self.ngrams[key] = []
            if continuation not in self.ngrams[key]:
                self.ngrams[key].append(continuation)
                if len(self.ngrams[key]) > 10:
                    self.ngrams[key] = self.ngrams[key][-10:]

    def get_candidates(self, context: List[int]) -> List[int]:
        """Get candidate continuations for the given context."""
        if len(context) < self.n:
            return []
        key = tuple(context[-self.n:])
        return self.ngrams.get(key, [])


class JacobiIteration(nn.Module):
    """Jacobi iteration for parallel token verification using level1 operators."""
    def __init__(self, hidden_size: int, vocab_size: int, max_iterations: int = 3):
        super().__init__()
        self.max_iterations = max_iterations
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(
        self,
        draft_tokens: torch.Tensor,
        logits_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Jacobi iteration for parallel verification.
        
        Args:
            draft_tokens: Initial draft token sequence (batch, draft_len)
            logits_fn: Function to compute logits from tokens
            
        Returns:
            verified_tokens: Final verified tokens
            accept_mask: Mask indicating accepted tokens
        """
        current = draft_tokens
        
        for _ in range(self.max_iterations):
            # Compute logits for all positions in parallel
            logits = logits_fn(current)
            
            # Greedy decode
            next_tokens = logits.argmax(dim=-1)
            
            # Check convergence
            if torch.all(next_tokens == current):
                break
            
            current = next_tokens
        
        # Compute acceptance mask (compare with original drafts)
        accept_mask = current == draft_tokens
        
        return current, accept_mask


class LookaheadAttention(nn.Module):
    """Attention with lookahead mask support using level1 operators."""
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

    def forward(
        self,
        x: torch.Tensor,
        lookahead_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary_emb(q, k)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = self.matmul(q, k.transpose(-2, -1)) * scale

        # Apply lookahead attention mask
        if lookahead_mask is not None:
            attn_weights = attn_weights.masked_fill(~lookahead_mask, float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = self.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LookaheadMLP(nn.Module):
    """MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class LookaheadLayer(nn.Module):
    """Lookahead decoder layer using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LookaheadAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LookaheadMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, lookahead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, lookahead_mask)
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
    Lookahead decoding model.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - RotaryEmbedding from level1/embeddings/1_RotaryEmbedding
    - Swish/SiLU from level1/activations/7_Swish
    - Softmax from level1/activations/5_Softmax
    - MatMul from level1/matmul/1_MatMul
    
    Supports variants: W4-N5, W7-N7 (configs loaded from config_loader)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "W4-N5", operator_level: Optional[OperatorLevel] = None, **kwargs):
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
        hidden_size = kwargs.get('hidden_size', 4096)
        num_heads = kwargs.get('num_heads', 32)
        num_kv_heads = kwargs.get('num_kv_heads', 8)
        head_dim = kwargs.get('head_dim', 128)
        intermediate_size = kwargs.get('intermediate_size', 14336)
        num_layers = kwargs.get('num_layers', 32)
        vocab_size = kwargs.get('vocab_size', 128256)
        lookahead_level = kwargs.get('lookahead_level', kwargs.get('window_size', 5))
        guess_set_size = kwargs.get('guess_set_size', kwargs.get('ngram_size', 64))
        
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
        
        self.lookahead_level = lookahead_level
        self.guess_set_size = guess_set_size
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Layers
        self.layers = nn.ModuleList([
            LookaheadLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Lookahead components
        self.ngram_pool = NgramPool()
        self.jacobi = JacobiIteration(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        lookahead_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, lookahead_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
