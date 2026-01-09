import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Token Type Embedding (Segment Embedding)
    
    Used by: BERT, RoBERTa, ALBERT (for sentence pair tasks)
    
    Embeds token type IDs (e.g., 0 for sentence A, 1 for sentence B).
    Used in encoder models to distinguish between segments.
    
    Shapes:
        Input: (batch_size, seq_length) - token type IDs
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, num_token_types: int = 2, hidden_size: int = 768):
        """
        Initialize Token Type Embedding.
        
        Args:
            num_token_types: Number of token types (default 2 for BERT)
            hidden_size: Embedding dimension
        """
        super(Model, self).__init__()
        self.token_type_embeddings = nn.Embedding(num_token_types, hidden_size)
    
    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up token type embeddings.
        
        Args:
            token_type_ids: Token type IDs (batch_size, seq_length)
            
        Returns:
            Token type embeddings (batch_size, seq_length, hidden_size)
        """
        return self.token_type_embeddings(token_type_ids)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
seq_length = 512
num_token_types = 2
hidden_size = 768

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Token type IDs: 0 for first segment, 1 for second segment
    token_type_ids = torch.randint(0, num_token_types, (batch_size, seq_length), device='cuda')
    return [token_type_ids]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_token_types, hidden_size]

