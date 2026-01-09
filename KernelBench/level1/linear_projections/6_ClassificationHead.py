import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Classification Head
    
    Used by: BERT, RoBERTa, DeBERTa (sequence classification, token classification)
    
    Projects CLS token (or pooled output) to class logits.
    Includes optional dropout and intermediate dense layer.
    
    Shapes:
        Input: (batch_size, hidden_size) - pooled output
        Output: (batch_size, num_classes) - class logits
    """
    
    def __init__(self, hidden_size: int = 768, num_classes: int = 2,
                 dropout_prob: float = 0.1):
        """
        Initialize Classification Head.
        
        Args:
            hidden_size: Input hidden dimension
            num_classes: Number of output classes
            dropout_prob: Dropout probability
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits.
        
        Args:
            pooled_output: Pooled hidden state (batch_size, hidden_size)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
hidden_size = 768
num_classes = 2
dropout_prob = 0.1

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    pooled_output = torch.randn(batch_size, hidden_size, device='cuda')
    return [pooled_output]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_classes, dropout_prob]

