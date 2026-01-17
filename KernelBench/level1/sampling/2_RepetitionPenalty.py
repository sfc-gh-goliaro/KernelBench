import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Repetition Penalty (Standalone Logits Processor)

    Used by: Repetition control in generation

    Penalize repeated tokens by scaling their logits down.
    Tokens that appear in the input context have their logits divided
    (if positive) or multiplied (if negative) by the penalty factor.

    This is a standalone processor that can be composed with other
    sampling strategies. For combined sampling, use UnifiedSampler.

    Shapes:
        logits: (batch_size, vocab_size) input logits
        input_ids: (batch_size, context_len) context token IDs
        Output: (batch_size, vocab_size) penalized logits
    """

    def __init__(self, penalty: float = 1.2):
        """
        Initialize repetition penalty.

        Args:
            penalty: Penalty factor (> 1 penalizes repetition, < 1 encourages it)
        """
        super(Model, self).__init__()
        self.penalty = penalty

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply repetition penalty to logits.

        Args:
            logits: Logits tensor (batch_size, vocab_size)
            input_ids: Previous token IDs (batch_size, context_len)

        Returns:
            Penalized logits (batch_size, vocab_size)
        """
        batch_size, vocab_size = logits.shape
        penalized_logits = logits.clone()

        # Vectorized penalty application
        for b in range(batch_size):
            unique_tokens = input_ids[b].unique()

            # Positive logits: divide by penalty (reduces probability)
            pos_mask = penalized_logits[b, unique_tokens] > 0
            penalized_logits[b, unique_tokens[pos_mask]] /= self.penalty

            # Negative logits: multiply by penalty (also reduces probability)
            neg_mask = penalized_logits[b, unique_tokens] < 0
            penalized_logits[b, unique_tokens[neg_mask]] *= self.penalty

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
