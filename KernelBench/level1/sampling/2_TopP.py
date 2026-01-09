import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-P (Nucleus) Sampling
    
    Used by: All LLM inference
    
    Sample from the smallest set of tokens with cumulative probability >= p.
    This adapts the number of tokens considered based on the distribution.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size,) sampled token indices
    """
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        """
        Initialize Top-P sampling.
        
        Args:
            p: Cumulative probability threshold
            temperature: Sampling temperature
        """
        super(Model, self).__init__()
        self.p = p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using nucleus (top-p) filtering.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Sampled token indices of shape (batch_size,)
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Convert to cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff: keep tokens until cumulative prob > p
        # Shift cumulative probs right to include the token that crosses threshold
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Set removed tokens to -inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # Convert back to probabilities and sample
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_sorted_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Map back to original vocabulary indices
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        sampled_tokens = sorted_indices[batch_indices, sampled_sorted_idx]
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
p = 0.9

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [p]

