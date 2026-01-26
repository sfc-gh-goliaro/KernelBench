"""
TIES-Merging Model

Implements TIES-Merging for model merging:
- Trim: Remove small-magnitude parameters
- Elect: Resolve sign conflicts
- Sum: Merge remaining parameters

Variants from Table 5:
- TIES-Llama-3.1-8B: TIES merging for Llama models

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

# Import level1 operators
# Operators that need wrapping
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
# Operators used directly
from ..level1.activations._7_Swish import Model as Swish
from ..level1.matmul._1_MatMul import Model as MatMul


# ============================================================================
# Model Variants from Table 5
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-8B": {
        "github_repo": "https://github.com/prateeky2806/ties-merging",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "num_layers": 32,
        "vocab_size": 128256,
        "trim_ratio": 0.2,
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


# ============================================================================
# TIES-Merging Operations
# ============================================================================

class TIESMerger:
    """TIES-Merging algorithm implementation."""
    
    @staticmethod
    def compute_task_vector(finetuned: torch.Tensor, pretrained: torch.Tensor) -> torch.Tensor:
        """Compute task vector (difference from pretrained)."""
        return finetuned - pretrained
    
    @staticmethod
    def trim(task_vector: torch.Tensor, trim_ratio: float) -> torch.Tensor:
        """Trim: Remove small-magnitude parameters."""
        # Calculate threshold based on magnitude
        magnitudes = task_vector.abs()
        threshold = torch.quantile(magnitudes.flatten(), trim_ratio)
        
        # Zero out values below threshold
        mask = magnitudes >= threshold
        return task_vector * mask.float()
    
    @staticmethod
    def elect_sign(task_vectors: List[torch.Tensor]) -> torch.Tensor:
        """Elect: Resolve sign conflicts by majority vote."""
        # Stack task vectors: (num_models, ...)
        stacked = torch.stack(task_vectors)
        
        # Count positive and negative signs
        positive_count = (stacked > 0).sum(dim=0).float()
        negative_count = (stacked < 0).sum(dim=0).float()
        
        # Majority sign: +1 if more positive, -1 if more negative
        majority_sign = torch.where(
            positive_count >= negative_count,
            torch.ones_like(positive_count),
            -torch.ones_like(positive_count)
        )
        
        return majority_sign
    
    @staticmethod
    def disjoint_merge(
        task_vectors: List[torch.Tensor],
        majority_sign: torch.Tensor,
    ) -> torch.Tensor:
        """Merge task vectors using disjoint selection based on majority sign."""
        # Stack and filter by majority sign
        stacked = torch.stack(task_vectors)
        
        # Keep only values matching majority sign
        sign_mask = (stacked.sign() == majority_sign.unsqueeze(0)).float()
        filtered = stacked * sign_mask
        
        # Average non-zero contributions
        count = sign_mask.sum(dim=0).clamp(min=1)
        merged = filtered.sum(dim=0) / count
        
        return merged
    
    @classmethod
    def merge(
        cls,
        pretrained: Dict[str, torch.Tensor],
        finetuned_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2,
        scaling_factor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Full TIES-Merging algorithm."""
        merged = {}
        
        for key in pretrained.keys():
            pretrained_param = pretrained[key]
            
            # Compute task vectors
            task_vectors = [
                cls.compute_task_vector(model[key], pretrained_param)
                for model in finetuned_models
            ]
            
            # Trim
            trimmed = [cls.trim(tv, trim_ratio) for tv in task_vectors]
            
            # Elect sign
            majority_sign = cls.elect_sign(trimmed)
            
            # Disjoint merge
            merged_tv = cls.disjoint_merge(trimmed, majority_sign)
            
            # Apply scaling and add back to pretrained
            merged[key] = pretrained_param + scaling_factor * merged_tv
        
        return merged


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class TIESLinear(nn.Module):
    """Linear layer supporting TIES merging."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Store pretrained weight for task vector computation
        self.register_buffer('pretrained_weight', self.weight.clone())
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
    
    def get_task_vector(self) -> torch.Tensor:
        return self.weight - self.pretrained_weight


class MergeableMLP(nn.Module):
    """MLP supporting TIES merging using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = TIESLinear(hidden_size, intermediate_size)
        self.up_proj = TIESLinear(hidden_size, intermediate_size)
        self.down_proj = TIESLinear(intermediate_size, hidden_size)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class MergeableBlock(nn.Module):
    """Transformer block supporting TIES merging using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        
        # Simplified attention
        self.qkv = TIESLinear(hidden_size, hidden_size * 3)
        self.o_proj = TIESLinear(hidden_size, hidden_size)
        
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MergeableMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        
        # Self attention
        residual = x
        x = self.input_layernorm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (hidden ** 0.5), dim=-1)
        x = torch.bmm(attn, v)
        x = self.o_proj(x)
        x = residual + x
        
        # MLP
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
    TIES-Merging model supporting efficient model merging.
    
    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
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
        trim_ratio: float = 0.2,
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
        
        self.trim_ratio = trim_ratio
        self.merger = TIESMerger()
        
        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Layers
        self.layers = nn.ModuleList([
            MergeableBlock(hidden_size, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.lm_head = TIESLinear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


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
        'intermediate_size': 14336,
        'num_layers': 8,
        'vocab_size': vocab_size,
        'trim_ratio': 0.2,
    }]
