import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Draft-Target Coordination for Speculative Decoding
    
    Used by: vLLM, SGLang speculative decoding
    
    Coordinates between draft model predictions and target model verification.
    Aligns draft tokens with target model's acceptance decisions.
    
    Shapes:
        Draft logits: (batch_size, num_draft_tokens, vocab_size)
        Target logits: (batch_size, num_draft_tokens + 1, vocab_size)
        Output: (batch_size, num_accepted + 1) - accepted tokens + bonus token
    """
    
    def __init__(self, vocab_size: int = 128256):
        """
        Initialize Draft-Target Coordination.
        
        Args:
            vocab_size: Vocabulary size
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
    
    def forward(self, draft_probs: torch.Tensor, target_probs: torch.Tensor,
                draft_tokens: torch.Tensor) -> tuple:
        """
        Coordinate draft and target for speculative acceptance.
        
        Uses rejection sampling to decide which draft tokens to accept.
        
        Args:
            draft_probs: Draft model probabilities (batch_size, num_draft, vocab_size)
            target_probs: Target model probabilities (batch_size, num_draft+1, vocab_size)
            draft_tokens: Tokens predicted by draft model (batch_size, num_draft)
            
        Returns:
            Tuple of (accepted_tokens, num_accepted):
                accepted_tokens: Final accepted token sequence
                num_accepted: Number of accepted draft tokens per sequence
        """
        batch_size, num_draft, _ = draft_probs.shape
        device = draft_probs.device
        
        # Get probabilities for draft tokens
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        pos_indices = torch.arange(num_draft, device=device).unsqueeze(0)
        
        # p_draft(x) and p_target(x) for each draft token
        p_draft = draft_probs[batch_indices, pos_indices, draft_tokens]
        p_target = target_probs[batch_indices, pos_indices, draft_tokens]
        
        # Acceptance probability: min(1, p_target / p_draft)
        acceptance_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)
        
        # Sample acceptance decisions
        uniform_samples = torch.rand_like(acceptance_prob)
        accepted = uniform_samples < acceptance_prob
        
        # Find first rejection point (or accept all)
        # Use cumulative product to find acceptance streak
        accept_cumsum = accepted.cumprod(dim=1)
        num_accepted = accept_cumsum.sum(dim=1)  # (batch_size,)
        
        # For rejected position, sample from modified distribution
        # p_modified = max(0, p_target - p_draft) (normalized)
        # This is simplified - just use target distribution
        
        # Build final token sequence
        max_accepted = num_accepted.max().item()
        output_length = max_accepted + 1  # accepted + bonus token
        
        # Gather accepted tokens + sample bonus token from target
        accepted_tokens = torch.zeros(batch_size, output_length, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            n = num_accepted[b].item()
            if n > 0:
                accepted_tokens[b, :n] = draft_tokens[b, :n]
            # Sample bonus token from target at position n
            bonus_token = torch.multinomial(target_probs[b, n], 1)
            accepted_tokens[b, n] = bonus_token
        
        return accepted_tokens, num_accepted


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 32
num_draft_tokens = 5
vocab_size = 128256

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    # Softmax probabilities
    draft_logits = torch.randn(batch_size, num_draft_tokens, vocab_size, device='cuda')
    target_logits = torch.randn(batch_size, num_draft_tokens + 1, vocab_size, device='cuda')
    
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    
    draft_tokens = torch.randint(0, vocab_size, (batch_size, num_draft_tokens), device='cuda')
    
    return [draft_probs, target_probs, draft_tokens]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [vocab_size]

