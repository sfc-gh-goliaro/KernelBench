"""
Jina Reranker Cross-Encoder Model

A cross-encoder model for semantic reranking:
- BERT-style encoder (bidirectional attention)
- Query-document pair encoding
- CLS pooling for classification
- Binary classification head for relevance scoring

Reference: Jina-Reranker-v2, BGE-Reranker-v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Embeddings(nn.Module):
    """BERT-style embeddings with word, position, and token type."""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (bidirectional)."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = hidden_states.shape

        q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with GELU."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """BERT-style encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.attention_norm = LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ffn_norm = LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attn_output)

        # FFN
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_norm(hidden_states + ffn_output)

        return hidden_states


class CLSPooler(nn.Module):
    """Pool CLS token representation."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        return self.activation(pooled)


class ClassificationHead(nn.Module):
    """Classification head for relevance scoring."""
    def __init__(self, hidden_size: int, num_labels: int = 1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class Model(nn.Module):
    """Jina Reranker Cross-Encoder Model."""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        num_labels: int = 1,
    ):
        super().__init__()

        self.embeddings = Embeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size
        )

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.pooler = CLSPooler(hidden_size)
        self.classifier = ClassificationHead(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Embed
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # Create attention mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
        else:
            extended_mask = None

        # Encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_mask)

        # Pool CLS token
        pooled = self.pooler(hidden_states)

        # Classify
        logits = self.classifier(pooled)

        # Return score (sigmoid for relevance probability)
        return torch.sigmoid(logits).squeeze(-1)


# Configuration (reduced for benchmarking)
batch_size = 32
sequence_length = 512  # Query + Document combined
vocab_size = 30522
hidden_size = 768
num_layers = 12
num_heads = 12
intermediate_size = 3072


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    # Token types: 0 for query, 1 for document
    # Assume query is first 64 tokens, document is rest
    token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    token_type_ids[:, 64:] = 1
    return [input_ids, attention_mask, token_type_ids]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'intermediate_size': intermediate_size,
        'max_position_embeddings': sequence_length,
    }]

