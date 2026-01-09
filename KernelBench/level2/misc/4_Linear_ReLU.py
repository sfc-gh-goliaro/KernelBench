import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Linear + ReLU
    
    Used by: All FFN/MLP layers
    
    Linear (GEMM) + Add + ReLU: standard MLP layer.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


batch_size, seq_len, in_features, out_features = 32, 512, 768, 3072

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features]

