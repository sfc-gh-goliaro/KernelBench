import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Speculative Verification for Speculative Decoding.
    
    Verifies draft tokens against the target model's predictions,
    accepting tokens that match and rejecting divergent continuations.
    
    Based on: "Fast Inference from Transformers via Speculative Decoding"
    """
    def __init__(self, vocab_size, temperature=1.0, top_p=0.9):
        """
        :param vocab_size: Size of vocabulary
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling threshold
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_p = top_p
        
    def nucleus_sample(self, logits):
        """Apply nucleus (top-p) sampling."""
        probs = F.softmax(logits / self.temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumsum_probs - sorted_probs > self.top_p
        sorted_probs[sorted_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Sample
        samples = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_indices, -1, samples).squeeze(-1)
    
    def forward(self, draft_tokens, draft_probs, target_logits):
        """
        Verify draft tokens against target model.
        
        Uses rejection sampling to accept tokens where draft matches target
        distribution well enough.
        
        :param draft_tokens: Proposed tokens from draft model (batch, num_draft)
        :param draft_probs: Probabilities of draft tokens (batch, num_draft)
        :param target_logits: Logits from target model (batch, num_draft, vocab_size)
        :return: Tuple of (accepted_tokens, num_accepted, replacement_tokens)
        """
        batch_size, num_draft = draft_tokens.shape
        device = draft_tokens.device
        
        # Get target probabilities
        target_probs = F.softmax(target_logits / self.temperature, dim=-1)
        
        # Get target probability for draft tokens
        target_probs_for_draft = torch.gather(
            target_probs, 
            dim=-1, 
            index=draft_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_draft)
        
        # Compute acceptance probability: min(1, p_target / p_draft)
        acceptance_prob = torch.clamp(
            target_probs_for_draft / (draft_probs + 1e-10), 
            max=1.0
        )
        
        # Random uniform for rejection sampling
        uniform = torch.rand_like(acceptance_prob)
        
        # Accept if uniform < acceptance_prob
        accepted_mask = uniform < acceptance_prob  # (batch, num_draft)
        
        # Find first rejection point for each sequence
        # After first rejection, all subsequent tokens are also rejected
        not_accepted = ~accepted_mask
        # Cumulative max to propagate first rejection
        rejection_point = not_accepted.cummax(dim=-1).values
        
        # Final accepted mask (reject everything after first rejection)
        final_accepted = ~rejection_point  # (batch, num_draft)
        
        # Count accepted tokens per sequence
        num_accepted = final_accepted.sum(dim=-1)  # (batch,)
        
        # Get replacement token for first rejected position
        # Use modified sampling: sample from (target - draft)^+ normalized
        first_rejected_idx = num_accepted.clamp(max=num_draft - 1)
        
        # Get logits at rejection point
        batch_indices = torch.arange(batch_size, device=device)
        rejection_logits = target_logits[batch_indices, first_rejected_idx]  # (batch, vocab)
        
        # Sample replacement using adjusted distribution
        # p_adjusted = max(0, p_target - p_draft)
        rejection_target_probs = F.softmax(rejection_logits / self.temperature, dim=-1)
        
        # Get draft model probability at rejection point (would need draft logits)
        # For simplicity, sample from target distribution
        replacement_tokens = self.nucleus_sample(rejection_logits)
        
        # Construct output: accepted drafts + one sampled token
        accepted_tokens = draft_tokens.clone()
        
        # Replace tokens after rejection with padding (-1)
        for b in range(batch_size):
            n_acc = num_accepted[b].item()
            if n_acc < num_draft:
                accepted_tokens[b, n_acc] = replacement_tokens[b]
                if n_acc + 1 < num_draft:
                    accepted_tokens[b, n_acc + 1:] = -1  # Padding
        
        return accepted_tokens, num_accepted, replacement_tokens


# Test parameters
batch_size = 16
num_draft = 8
vocab_size = 32000

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    draft_tokens = torch.randint(0, vocab_size, (batch_size, num_draft))
    draft_probs = F.softmax(torch.randn(batch_size, num_draft), dim=-1)
    target_logits = torch.randn(batch_size, num_draft, vocab_size)
    return [draft_tokens, draft_probs, target_logits]

def get_init_inputs():
    return [vocab_size]

