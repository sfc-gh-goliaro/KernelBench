import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Task Vector (Model Merging)

    Used by: Task Arithmetic, Model Editing

    Computes task vectors by subtracting pretrained weights from fine-tuned
    weights. Task vectors can be added/subtracted to transfer capabilities.

    Shapes:
        pretrained_weights: (param_shape) base model weights
        finetuned_weights: (param_shape) fine-tuned model weights
        Output: (param_shape) task vector
    """

    def __init__(self, scaling_factor: float = 1.0):
        """
        Initialize task vector computation.

        Args:
            scaling_factor: Factor to scale the task vector
        """
        super(Model, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, pretrained_weights: torch.Tensor,
                finetuned_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute task vector.

        Args:
            pretrained_weights: Base pretrained model weights
            finetuned_weights: Fine-tuned model weights

        Returns:
            Task vector (finetuned - pretrained) * scaling_factor
        """
        task_vector = finetuned_weights - pretrained_weights
        return task_vector * self.scaling_factor


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    # Llama-3.1 8B: attention layer task vector extraction
    {"param_shape": (4096, 4096)},
    # Llama-3.1 70B: large model task vector computation
    {"param_shape": (8192, 8192)},
    # Vicuna 13B: MLP layer task vector for instruction tuning
    {"param_shape": (5120, 13824)},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("model_merging", "7_TaskVector")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    pretrained = DISTRIBUTIONS[dist_name]((*p["param_shape"]), dtype=dtype, device=device)
    return [pretrained, finetuned]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return [1.0]
