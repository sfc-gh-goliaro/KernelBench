import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-K Sampling
    
    Used by: All LLM inference
    
    Sample from the top-k highest probability tokens, redistributing
    probability mass among them.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size,) sampled token indices
    """
    
    def __init__(self, k: int = 50, temperature: float = 1.0):
        """
        Initialize Top-K sampling.
        
        Args:
            k: Number of top tokens to consider
            temperature: Sampling temperature
        """
        super(Model, self).__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using top-k filtering.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Sampled token indices of shape (batch_size,)
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Convert to probabilities
        probs = F.softmax(top_k_values, dim=-1)
        
        # Sample from the filtered distribution
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Map back to original vocabulary indices
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        sampled_tokens = top_k_indices[batch_indices, sampled_idx]
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
k = 50

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [k]

