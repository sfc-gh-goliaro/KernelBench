import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


PARAMETERS = [
    {"batch_size": 64, "vocab_size": 32000, "context_len": 512, "penalty": 1.2},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("sampling", "2_RepetitionPenalty")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    logits = DISTRIBUTIONS[dist_name]((p["batch_size"], p["vocab_size"]), dtype=dtype, device=device)
    return [logits, input_ids]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["penalty"]]
