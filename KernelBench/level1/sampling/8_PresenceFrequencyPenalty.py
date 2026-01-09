import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Presence and Frequency Penalty
    
    Used by: ChatGPT-style APIs
    
    OpenAI-style presence and frequency penalties for diverse generation.
    Presence penalty: flat penalty for any token that appears
    Frequency penalty: penalty proportional to token count
    
    Shapes:
        Input: (batch_size, vocab_size) logits, (batch_size, context_len) input_ids
        Output: (batch_size, vocab_size) penalized logits
    """
    
    def __init__(self, presence_penalty: float = 0.0, frequency_penalty: float = 0.0):
        """
        Initialize presence/frequency penalties.
        
        Args:
            presence_penalty: Flat penalty for present tokens
            frequency_penalty: Per-occurrence penalty
        """
        super(Model, self).__init__()
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
    
    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply presence and frequency penalties.
        
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            input_ids: Previous token IDs of shape (batch_size, context_len)
            
        Returns:
            Penalized logits of shape (batch_size, vocab_size)
        """
        batch_size, vocab_size = logits.shape
        
        # Count token frequencies in input
        token_counts = torch.zeros(batch_size, vocab_size, device=logits.device)
        for b in range(batch_size):
            unique, counts = input_ids[b].unique(return_counts=True)
            token_counts[b, unique] = counts.float()
        
        # Presence penalty: apply to any token that appears (count > 0)
        presence_mask = (token_counts > 0).float()
        
        # Frequency penalty: proportional to count
        # Final penalty = presence_penalty * present + frequency_penalty * count
        penalty = (self.presence_penalty * presence_mask + 
                  self.frequency_penalty * token_counts)
        
        # Subtract penalty from logits
        penalized_logits = logits - penalty
        
        return penalized_logits


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
vocab_size = 32000
context_len = 512
presence_penalty = 0.6
frequency_penalty = 0.6

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    logits = torch.randn(batch_size, vocab_size, device='cuda')
    input_ids = torch.randint(0, vocab_size, (batch_size, context_len), device='cuda')
    return [logits, input_ids]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [presence_penalty, frequency_penalty]

