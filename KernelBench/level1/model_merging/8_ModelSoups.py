import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
from typing import List

class Model(nn.Module):
    """
    Model Soups (Model Merging)

    Used by: WiSE-FT, Model Soups

    Averages weights from multiple fine-tuned checkpoints (model soup).
    Can use uniform averaging or greedy soup selection based on validation.

    Shapes:
        weights_list: List of (param_shape) weight tensors
        scores: Optional (num_models,) validation scores for weighted averaging
        Output: (param_shape) merged weights
    """

    def __init__(self, mode: str = 'uniform'):
        """
        Initialize model soups.

        Args:
            mode: 'uniform' for equal weights, 'weighted' for score-based
        """
        super(Model, self).__init__()
        self.mode = mode

    def forward(self, *weights_and_scores) -> torch.Tensor:
        """
        Merge multiple model weights.

        Args:
            *weights_and_scores: Variable number of weight tensors,
                                optionally followed by scores tensor

        Returns:
            Merged weights (param_shape)
        """
        # Check if last argument is scores (1D tensor with length matching num weights)
        if len(weights_and_scores) >= 2:
            potential_scores = weights_and_scores[-1]
            if potential_scores.dim() == 1 and len(potential_scores) == len(weights_and_scores) - 1:
                weights = weights_and_scores[:-1]
                scores = potential_scores
            else:
                weights = weights_and_scores
                scores = None
        else:
            weights = weights_and_scores
            scores = None

        num_models = len(weights)

        if self.mode == 'uniform' or scores is None:
            # Simple uniform averaging
            merged = sum(weights) / num_models
        else:
            # Weighted averaging based on scores
            # Normalize scores to sum to 1
            normalized_scores = torch.softmax(scores, dim=0)

            merged = torch.zeros_like(weights[0])
            for w, s in zip(weights, normalized_scores):
                merged += w * s

        return merged


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Llama-3.1 8B: model soup from 5 fine-tuned checkpoints
    {"param_shape": (4096, 4096), "num_models": 5},
    # Mistral 7B: greedy soup from 8 domain-adapted models
    {"param_shape": (14336, 4096), "num_models": 8},
    # Qwen2 7B: uniform soup from 3 instruction-tuned variants
    {"param_shape": (18944, 3584), "num_models": 3},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "8_ModelSoups")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    base = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    scores = DISTRIBUTIONS[dist_name]((p["num_models"]), dtype=dtype, device=device)
    return []

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return ['weighted']
