import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Temperature Scaling
    
    Used by: All LLM inference
    
    Scale logits by temperature before softmax to control randomness.
    Higher temperature = more random, lower = more deterministic.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size, vocab_size) scaled probabilities
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature scaling.
        
        Args:
            temperature: Scaling temperature (must be > 0)
        """
        super(Model, self).__init__()
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling and convert to probabilities.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Scaled probabilities of shape (batch_size, vocab_size)
        """
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
temperature = 0.7

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [temperature]

