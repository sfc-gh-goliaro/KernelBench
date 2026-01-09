import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Repetition Penalty
    
    Used by: Repetition control in generation
    
    Penalize repeated tokens by scaling their logits down.
    Tokens that appear in the input context have their logits divided
    by the penalty factor (if positive) or multiplied (if negative).
    
    Shapes:
        Input: (batch_size, vocab_size) logits, (batch_size, context_len) input_ids
        Output: (batch_size, vocab_size) penalized logits
    """
    
    def __init__(self, penalty: float = 1.2):
        """
        Initialize repetition penalty.
        
        Args:
            penalty: Penalty factor (> 1 penalizes, < 1 encourages)
        """
        super(Model, self).__init__()
        self.penalty = penalty
    
    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply repetition penalty to logits.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            input_ids: Previous token IDs of shape (batch_size, context_len)
            
        Returns:
            Penalized logits of shape (batch_size, vocab_size)
        """
        batch_size, vocab_size = logits.shape
        
        # Create a penalty mask
        penalty_mask = torch.zeros_like(logits)
        
        # Mark tokens that appear in input_ids
        for b in range(batch_size):
            unique_tokens = input_ids[b].unique()
            penalty_mask[b, unique_tokens] = 1.0
        
        # Apply penalty: divide positive logits, multiply negative logits
        # This ensures penalty always reduces the probability
        penalized_logits = logits.clone()
        
        # For positive logits: divide by penalty
        positive_mask = (logits > 0) & (penalty_mask > 0)
        penalized_logits[positive_mask] = logits[positive_mask] / self.penalty
        
        # For negative logits: multiply by penalty  
        negative_mask = (logits < 0) & (penalty_mask > 0)
        penalized_logits[negative_mask] = logits[negative_mask] * self.penalty
        
        return penalized_logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
context_len = 512
penalty = 1.2

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    input_ids = torch.randint(0, vocab_size, (batch_size, context_len), device='cuda')
    return [logits, input_ids]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [penalty]

