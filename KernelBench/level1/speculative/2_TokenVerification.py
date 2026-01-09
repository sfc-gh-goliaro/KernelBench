import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Token Verification (Speculative Decoding)
    
    Used by: All speculative decoding methods
    
    Accept/reject draft tokens by comparing draft vs target distributions
    with rejection sampling.
    
    Shapes:
        draft_probs: (batch, num_draft, vocab_size)
        target_probs: (batch, num_draft, vocab_size)
        draft_tokens: (batch, num_draft)
        Output: (batch,) number of accepted tokens per sequence
    """
    
    def __init__(self):
        """Initialize token verification."""
        super(Model, self).__init__()
    
    def forward(self, draft_probs: torch.Tensor, target_probs: torch.Tensor,
                draft_tokens: torch.Tensor) -> tuple:
        """
        Verify draft tokens using rejection sampling.
        
        Args:
            draft_probs: Draft model probabilities (batch, num_draft, vocab)
            target_probs: Target model probabilities (batch, num_draft, vocab)
            draft_tokens: Drafted token IDs (batch, num_draft)
            
        Returns:
            Tuple of (num_accepted, accepted_mask):
                num_accepted: (batch,) number of accepted tokens
                accepted_mask: (batch, num_draft) boolean mask of accepted tokens
        """
        batch_size, num_draft, vocab_size = draft_probs.shape
        device = draft_probs.device
        
        # Get probabilities for the drafted tokens
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        draft_indices = torch.arange(num_draft, device=device).unsqueeze(0)
        
        p_draft = draft_probs[batch_indices, draft_indices, draft_tokens]  # (batch, num_draft)
        p_target = target_probs[batch_indices, draft_indices, draft_tokens]  # (batch, num_draft)
        
        # Acceptance probability: min(1, p_target / p_draft)
        acceptance_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
        
        # Sample uniform random numbers
        u = torch.rand_like(acceptance_prob)
        
        # Accept if u < acceptance_prob
        accepted = u < acceptance_prob  # (batch, num_draft)
        
        # Find first rejection for each sequence (all after are rejected)
        # Use cumulative product trick: accepted only if all previous were accepted
        cumulative_accepted = accepted.cumprod(dim=1)
        
        # Count accepted tokens
        num_accepted = cumulative_accepted.sum(dim=1)  # (batch,)
        
        return num_accepted, cumulative_accepted


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64
num_draft = 5
vocab_size = 32000

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    draft_probs = F.softmax(torch.randn(batch_size, num_draft, vocab_size, device='cuda'), dim=-1)
    target_probs = F.softmax(torch.randn(batch_size, num_draft, vocab_size, device='cuda'), dim=-1)
    draft_tokens = torch.randint(0, vocab_size, (batch_size, num_draft), device='cuda')
    return [draft_probs, target_probs, draft_tokens]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []

