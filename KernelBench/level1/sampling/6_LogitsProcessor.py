import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Logits Processor
    
    Used by: All LLM inference
    
    Base logits processor that applies transformations before sampling.
    This implementation applies temperature scaling and optional top-k mask.
    
    Shapes:
        Input: (batch_size, vocab_size) logits
        Output: (batch_size, vocab_size) processed logits
    """
    
    def __init__(self, temperature: float = 1.0, top_k: int = 0):
        """
        Initialize logits processor.
        
        Args:
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
        """
        super(Model, self).__init__()
        self.temperature = temperature
        self.top_k = top_k
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Process logits before sampling.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Processed logits of shape (batch_size, vocab_size)
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Apply top-k filtering
        if self.top_k > 0:
            top_k_values = torch.topk(logits, self.top_k, dim=-1).values
            threshold = top_k_values[..., -1, None]
            logits = torch.where(logits < threshold, 
                                torch.full_like(logits, float('-inf')), logits)
        
        return logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    return [logits]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [0.7, 50]  # temperature=0.7, top_k=50

