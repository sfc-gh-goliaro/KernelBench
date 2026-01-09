import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Min-P Sampling
    
    Used by: Alternative sampling method
    
    Filter tokens with probability less than min_p * max_probability.
    This is an alternative to top-k/top-p that scales with the confidence.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size,) sampled token indices
    """
    
    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        """
        Initialize Min-P sampling.
        
        Args:
            min_p: Minimum probability threshold relative to max prob
            temperature: Sampling temperature
        """
        super(Model, self).__init__()
        self.min_p = min_p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using min-p filtering.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Sampled token indices of shape (batch_size,)
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute threshold: min_p * max_prob for each sample
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = self.min_p * max_probs
        
        # Filter out tokens below threshold
        filtered_probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        
        # Sample
        sampled_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
min_p = 0.05

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [min_p]

