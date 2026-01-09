"""
DeBERTa-v3 Encoder-Only Transformer

A modern encoder-only transformer implementing DeBERTa-v3 features:
- Disentangled attention with separate content and position embeddings
- Enhanced mask decoder (EMD)
- Relative position bias
- GELU activation

Reference: DeBERTa-v3-large
- Hidden: 1024, Heads: 16, FFN: 4096, Layers: 24
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


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled Self-Attention from DeBERTa.
    
    Computes attention with separate content-to-content, content-to-position,
    and position-to-content attention scores.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 512,
        pos_att_type: str = "c2p|p2c",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pos_att_type = pos_att_type

        # Content projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Position projections (for disentangled attention)
        self.pos_q_proj = nn.Linear(hidden_size, hidden_size)
        self.pos_k_proj = nn.Linear(hidden_size, hidden_size)

        # Relative position embedding
        self.rel_pos_embed = nn.Embedding(max_position_embeddings * 2, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Content projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Relative position indices
        position_ids = torch.arange(seq_len, device=hidden_states.device)
        rel_pos = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        rel_pos = rel_pos + self.max_position_embeddings  # Shift to positive
        rel_pos = rel_pos.clamp(0, self.max_position_embeddings * 2 - 1)

        # Position embeddings
        rel_pos_embed = self.rel_pos_embed(rel_pos)  # (seq, seq, hidden)

        # Position projections
        pos_q = self.pos_q_proj(rel_pos_embed)
        pos_k = self.pos_k_proj(rel_pos_embed)

        pos_q = pos_q.view(seq_len, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        pos_k = pos_k.view(seq_len, seq_len, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        # Content-to-content attention
        c2c_attn = torch.matmul(q, k.transpose(-2, -1))

        # Content-to-position attention (c2p)
        # q @ pos_k^T
        c2p_attn = torch.einsum('bhid,hjid->bhij', q, pos_k)

        # Position-to-content attention (p2c)
        # pos_q @ k^T
        p2c_attn = torch.einsum('hijd,bhjd->bhij', pos_q, k)

        # Combine attention scores
        attn_scores = c2c_attn + c2p_attn + p2c_attn
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(attn_output)


class DeBERTaIntermediate(nn.Module):
    """DeBERTa FFN intermediate layer."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(x))


class DeBERTaOutput(nn.Module):
    """DeBERTa FFN output layer."""
    def __init__(self, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(x))


class DeBERTaLayer(nn.Module):
    """Single DeBERTa encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = DisentangledSelfAttention(hidden_size, num_heads)
        self.attention_norm = LayerNorm(hidden_size)
        self.intermediate = DeBERTaIntermediate(hidden_size, intermediate_size)
        self.output = DeBERTaOutput(intermediate_size, hidden_size)
        self.output_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(x, attention_mask)
        x = self.attention_norm(x + attn_output)

        # FFN with residual
        ffn_output = self.output(self.intermediate(x))
        x = self.output_norm(x + ffn_output)

        return x


class DeBERTaEmbeddings(nn.Module):
    """DeBERTa embeddings (word + position for EMD)."""
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class DeBERTaPooler(nn.Module):
    """Pool CLS token for classification."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0]
        return torch.tanh(self.dense(cls_token))


class Model(nn.Module):
    """DeBERTa-v3 style encoder-only transformer."""
    def __init__(
        self,
        vocab_size: int = 128100,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.embeddings = DeBERTaEmbeddings(vocab_size, hidden_size, max_position_embeddings)

        self.layers = nn.ModuleList([
            DeBERTaLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

        self.pooler = DeBERTaPooler(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Create attention mask if needed
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_mask)

        pooled_output = self.pooler(hidden_states)

        return pooled_output


# Configuration (reduced for benchmarking)
batch_size = 8
sequence_length = 512
vocab_size = 128100
hidden_size = 768
num_layers = 6
num_heads = 12
intermediate_size = 3072


def get_inputs():
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    return [input_ids, attention_mask]


def get_init_inputs():
    return [{
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'intermediate_size': intermediate_size,
        'max_position_embeddings': sequence_length,
    }]

