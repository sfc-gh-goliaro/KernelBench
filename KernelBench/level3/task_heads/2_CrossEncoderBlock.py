import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BERTSelfAttention(nn.Module):
    """BERT-style self-attention for cross-encoder."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)


class BERTLayer(nn.Module):
    """Single BERT encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = BERTSelfAttention(hidden_size, num_heads, dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, attention_mask)
        attn_output = self.dropout(self.attention_output(attn_output))
        x = self.attention_norm(x + attn_output)
        
        # FFN
        ffn_output = F.gelu(self.intermediate(x))
        ffn_output = self.dropout(self.output(ffn_output))
        x = self.output_norm(x + ffn_output)
        
        return x


class Model(nn.Module):
    """
    Cross-Encoder Block (Reranker)
    
    A BERT-based block for cross-encoder reranking models.
    Used by: BGE-Reranker, Jina-Reranker, ms-marco-MiniLM
    
    Architecture:
        [CLS] query [SEP] document [SEP] -> BERT Layers -> CLS pooling
                                         -> Classification head -> Relevance score
    
    Key features:
    - Joint encoding of query-document pairs
    - CLS token pooling for classification
    - Direct relevance scoring
    
    Cross-encoders are more accurate than bi-encoders but slower
    since they can't pre-compute document embeddings.
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 num_labels: int = 1, dropout: float = 0.1):
        super().__init__()
        
        # Single BERT layer (in practice, stack multiple)
        self.encoder_layer = BERTLayer(hidden_size, num_heads, intermediate_size, dropout)
        
        # Classification head
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            hidden_states: Token embeddings (batch, seq_len, hidden_size)
                          Input should be [CLS] query [SEP] document [SEP]
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            Relevance scores (batch, num_labels)
        """
        # Create attention mask for BERT (0 for attend, -inf for ignore)
        if attention_mask is not None:
            extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        else:
            extended_mask = None
        
        # Encode
        hidden_states = self.encoder_layer(hidden_states, extended_mask)
        
        # CLS pooling
        cls_output = hidden_states[:, 0]  # First token is [CLS]
        pooled_output = self.pooler(cls_output)
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        
        return logits


# Benchmark configuration (BERT-base for reranking)
batch_size = 32
seq_len = 512  # Query + Document
hidden_size = 768
num_heads = 12
intermediate_size = 3072
dropout = 0.0

def get_inputs():
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    return [hidden_states, attention_mask]

def get_init_inputs():
    return [hidden_size, num_heads, intermediate_size]

