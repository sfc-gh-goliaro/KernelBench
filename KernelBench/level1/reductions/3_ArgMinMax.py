import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Argmin or Argmax over a specified dimension.
    Both operations use identical reduction kernels that track value and index,
    differing only in the comparison operator.
    """
    def __init__(self, dim: int, mode: str = "max"):
        """
        Initializes the model with the dimension and mode for arg reduction.

        Args:
            dim (int): The dimension to perform arg reduction over.
            mode (str): Either "min" or "max" to select argmin or argmax.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmin or argmax over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with indices of min/max values.
        """
        if self.mode == "max":
            return torch.argmax(x, dim=self.dim)
        else:
            return torch.argmin(x, dim=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1, "max"]  # dim=1, mode="max"
