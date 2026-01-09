import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Learned Position Embedding
    
    Used by: BERT, GPT-2, ViT
    
    Learned absolute position embeddings added to token embeddings.
    Each position has a unique learned embedding vector.
    
    Shapes:
        Input: (batch_size, seq_len, hidden_size)
        Output: (batch_size, seq_len, hidden_size)
    """
    
    def __init__(self, max_seq_len: int, hidden_size: int):
        """
        Initialize learned position embeddings.
        
        Args:
            max_seq_len: Maximum sequence length
            hidden_size: Embedding dimension
        """
        super(Model, self).__init__()
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Add position embeddings to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Optional position indices (batch_size, seq_len)
            
        Returns:
            Tensor with position embeddings added
        """
        seq_len = x.shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        position_embeddings = self.position_embedding(position_ids)
        
        return x + position_embeddings


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096
max_seq_len = 8192

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    x = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [x]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [max_seq_len, hidden_size]

