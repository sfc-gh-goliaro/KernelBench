import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Score Head for Reward Models
    
    Used by: Reward models (Skywork-Reward, ArmoRM, Nemotron-RM)
    
    Projects the last token's hidden state to a scalar score.
    Used in RLHF pipelines for preference learning.
    
    Shapes:
        Input: (batch_size, hidden_size) - last token hidden state
        Output: (batch_size, 1) or (batch_size,) - scalar scores
    """
    
    def __init__(self, hidden_size: int = 4096, num_labels: int = 1):
        """
        Initialize Score Head.
        
        Args:
            hidden_size: Input hidden dimension
            num_labels: Number of output scores (1 for single reward)
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Score projection
        self.score = nn.Linear(hidden_size, num_labels, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute reward score from hidden states.
        
        Args:
            hidden_states: Last token hidden state (batch_size, hidden_size)
            
        Returns:
            Scores (batch_size, num_labels)
        """
        scores = self.score(hidden_states)
        if self.num_labels == 1:
            scores = scores.squeeze(-1)
        return scores


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
hidden_size = 4096
num_labels = 1

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Last token hidden states
    hidden_states = torch.randn(batch_size, hidden_size, device='cuda')
    return [hidden_states]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size, num_labels]

