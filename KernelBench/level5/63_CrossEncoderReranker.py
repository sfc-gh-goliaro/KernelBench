import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Cross-Encoder for Document Reranking.
    
    Jointly encodes query and document pairs for relevance scoring.
    Used in Jina Reranker and similar retrieval systems.
    
    Based on: BERT Cross-Encoder and Jina Reranker architectures
    """
    def __init__(self, hidden_dim, num_heads, num_layers, vocab_size=32000, 
                 max_length=512, dropout=0.1):
        """
        :param hidden_dim: Hidden dimension
        :param num_heads: Number of attention heads
        :param num_layers: Number of transformer layers
        :param vocab_size: Vocabulary size
        :param max_length: Maximum sequence length
        :param dropout: Dropout rate
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)
        self.segment_embed = nn.Embedding(2, hidden_dim)  # Query vs Document
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Final norm and classifier
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional: cross-attention layers for late interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        Compute relevance score for query-document pair.
        
        :param input_ids: Token IDs [CLS] query [SEP] document [SEP]
                         (batch, seq_len)
        :param segment_ids: Segment IDs (0 for query, 1 for document)
                           (batch, seq_len)
        :param attention_mask: Attention mask (batch, seq_len)
        :return: Relevance scores (batch, 1)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed[:, :seq_len]
        seg_emb = self.segment_embed(segment_ids)
        
        # Combine embeddings
        x = token_emb + pos_emb + seg_emb
        
        # Create attention mask if needed
        if attention_mask is not None:
            # Convert to key_padding_mask (True = masked out)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Final norm
        x = self.norm(x)
        
        # Get CLS token representation
        cls_output = x[:, 0]
        
        # Compute relevance score
        score = self.classifier(cls_output)
        
        return score
    
    def rerank(self, query_doc_pairs, return_scores=True):
        """
        Rerank multiple query-document pairs.
        
        :param query_doc_pairs: List of (input_ids, segment_ids, attention_mask) tuples
        :param return_scores: Whether to return scores or just rankings
        :return: Sorted indices and optionally scores
        """
        scores = []
        
        for input_ids, segment_ids, attention_mask in query_doc_pairs:
            score = self.forward(input_ids, segment_ids, attention_mask)
            scores.append(score)
        
        scores = torch.cat(scores, dim=0)
        sorted_indices = torch.argsort(scores.squeeze(-1), descending=True)
        
        if return_scores:
            return sorted_indices, scores
        return sorted_indices


# Test parameters
batch_size = 16
seq_len = 256
hidden_dim = 768
num_heads = 12
num_layers = 12
vocab_size = 32000

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    segment_ids[:, 64:] = 1  # First 64 tokens are query, rest is document
    attention_mask = torch.ones(batch_size, seq_len)
    return [input_ids, segment_ids, attention_mask]

def get_init_inputs():
    return [hidden_dim, num_heads, num_layers, vocab_size]

