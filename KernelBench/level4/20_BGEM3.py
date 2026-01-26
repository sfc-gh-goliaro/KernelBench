"""
BGE-M3 Embedding Model

Implements BGE-M3 multi-functional text embedding:
- Dense embedding (for semantic similarity)
- Sparse embedding (for lexical matching)
- Multi-vector ColBERT-style embedding

Variants from Table 5:
- BGE-M3: Multi-lingual, multi-task embedding model

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

# Import level1 operators (used directly - no wrapping needed)
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.activations._8_GELU import Model as GELU
from ..level1.matmul._1_MatMul import Model as MatMul
from ..level1.pooling._9_MeanPooling import Model as MeanPooling


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "M3": "BAAI/bge-m3",
}


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class BertSelfAttention(nn.Module):
    """BERT-style self-attention using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        
        self.matmul = MatMul()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = self.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        out = self.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        
        return self.dense(out)


class BertMLP(nn.Module):
    """BERT-style MLP using level1 operators."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.gelu = GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_output(self.gelu(self.dense(x)))


class BertLayer(nn.Module):
    """BERT encoder layer using level1 operators."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = BertSelfAttention(hidden_size, num_heads)
        self.attention_norm = LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.output_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention with residual
        attn_out = self.attention(x, attention_mask)
        x = self.attention_norm(x + attn_out)
        
        # MLP with residual
        mlp_out = self.mlp(x)
        x = self.output_norm(x + mlp_out)
        
        return x


class SparseEmbeddingHead(nn.Module):
    """Sparse embedding head for lexical matching."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Compute token weights
        token_weights = self.linear(hidden_states).squeeze(-1)
        token_weights = torch.relu(token_weights)
        
        # Create sparse representation
        batch_size, seq_len = input_ids.shape
        sparse_emb = torch.zeros(batch_size, self.vocab_size, device=hidden_states.device)
        
        # Scatter token weights to vocabulary positions
        sparse_emb.scatter_add_(1, input_ids, token_weights)
        
        return sparse_emb


class ColBERTHead(nn.Module):
    """ColBERT-style multi-vector head."""
    def __init__(self, hidden_size: int, output_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(hidden_states), dim=-1)


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    BGE-M3 multi-functional embedding model.
    
    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GELU from level1/activations/8_GELU
    - MatMul from level1/matmul/1_MatMul
    - MeanPooling from level1/pooling/9_MeanPooling
    
    Supports variants: M3 (configs loaded from HuggingFace)
    """
    
    VARIANTS = VARIANTS
    
    @classmethod
    def from_pretrained(cls, variant: str = "M3", operator_level: Optional[OperatorLevel] = None, **kwargs):
        """Create model with config loaded from HuggingFace."""
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
        hidden_size = kwargs.get('hidden_size', 1024)
        num_heads = kwargs.get('num_heads', 16)
        num_layers = kwargs.get('num_layers', 24)
        vocab_size = kwargs.get('vocab_size', 250002)
        intermediate_size = kwargs.get('intermediate_size', 4096)
        max_seq_len = kwargs.get('max_seq_len', 8192)
        colbert_dim = kwargs.get('colbert_dim', 128)
        
        if config is None:
            config = ModelConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=vocab_size,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
            )
        
        super().__init__()
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.embedding_norm = LayerNorm(hidden_size)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.dense_head = nn.Linear(hidden_size, hidden_size)  # For dense embeddings
        self.sparse_head = SparseEmbeddingHead(hidden_size, vocab_size)  # For sparse embeddings
        self.colbert_head = ColBERTHead(hidden_size, colbert_dim)  # For multi-vector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.embedding_norm(x)
        
        # Create attention mask
        if attention_mask is not None:
            extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        else:
            extended_mask = None
        
        # Encoder
        for layer in self.layers:
            x = layer(x, extended_mask)
        
        # Dense embedding (CLS token or mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (x * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            dense_embedding = sum_embeddings / sum_mask
        else:
            dense_embedding = x.mean(dim=1)
        
        dense_embedding = F.normalize(self.dense_head(dense_embedding), dim=-1)
        
        # Sparse embedding
        sparse_embedding = self.sparse_head(x, input_ids)
        
        # ColBERT multi-vector
        colbert_embedding = self.colbert_head(x)
        
        return {
            'dense': dense_embedding,
            'sparse': sparse_embedding,
            'colbert': colbert_embedding,
        }


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
sequence_length = 512
vocab_size = 250002


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    return [input_ids, attention_mask]


def get_init_inputs():
    return [{
        'hidden_size': 1024,
        'num_heads': 16,
        'num_layers': 8,
        'vocab_size': vocab_size,
        'intermediate_size': 4096,
    }]
