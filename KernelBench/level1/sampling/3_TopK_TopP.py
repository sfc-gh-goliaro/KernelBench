import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Combined Top-K and Top-P Sampling
    
    Used by: vLLM, SGLang default sampler
    
    Apply both top-k and top-p filtering before sampling for
    more controlled generation.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size,) sampled token indices
    """
    
    def __init__(self, k: int = 50, p: float = 0.9, temperature: float = 1.0):
        """
        Initialize combined Top-K/Top-P sampling.
        
        Args:
            k: Number of top tokens to consider
            p: Cumulative probability threshold
            temperature: Sampling temperature
        """
        super(Model, self).__init__()
        self.k = k
        self.p = p
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using combined top-k and top-p filtering.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Sampled token indices of shape (batch_size,)
        """
        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # First apply top-k filtering
        if self.k > 0:
            top_k_threshold = torch.topk(logits, self.k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < top_k_threshold, 
                                torch.full_like(logits, float('-inf')), logits)
        
        # Then apply top-p filtering
        if self.p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Create mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > self.p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter mask back to original order
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return sampled_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
k = 50
p = 0.9

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [k, p]

