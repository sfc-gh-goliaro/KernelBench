import os
import sys
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
