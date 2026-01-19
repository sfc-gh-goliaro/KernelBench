import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Jacobi Iteration (Speculative Decoding)

    Used by: Jacobi Decoding, CLLMs (Consistency LLMs)

    Parallel token prediction using Jacobi iteration. Initializes n-gram
    with random/draft tokens and iteratively refines all positions until
    convergence, enabling parallel decoding.

    Shapes:
        hidden_states: (batch, n_positions, hidden_size)
        initial_tokens: (batch, n_positions) initial token guesses
        Output: (batch, n_positions) refined token IDs
    """

    def __init__(self, hidden_size: int, vocab_size: int, max_iterations: int = 10):
        """
        Initialize Jacobi iteration decoder.

        Args:
            hidden_size: Model hidden dimension
            vocab_size: Vocabulary size
            max_iterations: Maximum refinement iterations
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_iterations = max_iterations

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_size)

        # Refinement layer (simplified - real impl would use full transformer)
        self.refine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, initial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Perform Jacobi iteration to refine token predictions.

        Args:
            hidden_states: Context hidden states (batch, n_positions, hidden_size)
            initial_tokens: Initial token guesses (batch, n_positions)

        Returns:
            Refined token IDs (batch, n_positions)
        """
        batch_size, n_positions, _ = hidden_states.shape
        current_tokens = initial_tokens.clone()

        for iteration in range(self.max_iterations):
            # Get embeddings for current tokens
            token_embeds = self.token_embed(current_tokens)

            # Combine with hidden states and refine
            combined = torch.cat([hidden_states, token_embeds], dim=-1)
            refined = self.refine(combined)

            # Get new token predictions
            logits = self.lm_head(refined)
            new_tokens = logits.argmax(dim=-1)

            # Check for convergence
            if torch.equal(new_tokens, current_tokens):
                break

            current_tokens = new_tokens

        return current_tokens


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Low-latency: Llama-3.1-8B deep Jacobi window (16 positions)
    {"batch_size": 4, "n_positions": 16, "hidden_size": 4096, "vocab_size": 128256},
    # Low-latency: Llama-3.1-70B extended consistency decoding (12 positions)
    {"batch_size": 2, "n_positions": 12, "hidden_size": 8192, "vocab_size": 128256},
    # High-throughput: Llama-2-7B batched Jacobi (8 positions, large batch)
    {"batch_size": 64, "n_positions": 8, "hidden_size": 4096, "vocab_size": 32000},
    # High-throughput: Vicuna-7B high-volume consistency (6 positions)
    {"batch_size": 32, "n_positions": 6, "hidden_size": 4096, "vocab_size": 32000},
    # Balanced: Llama-3.1-8B standard Jacobi
    {"batch_size": 16, "n_positions": 10, "hidden_size": 4096, "vocab_size": 128256},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("speculative", "6_JacobiIteration")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    hidden_states = DISTRIBUTIONS[dist_name]((p["batch_size"], p["n_positions"], p["hidden_size"]), dtype=dtype, device=device)
    return [hidden_states, initial_tokens]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["hidden_size"], p["vocab_size"]]
