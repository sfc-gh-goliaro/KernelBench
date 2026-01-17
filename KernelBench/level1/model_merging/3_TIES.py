import torch
import torch.nn as nn

class Model(nn.Module):
    """
    TIES Merging
    
    Used by: TIES merging
    
    Trim low-magnitude, Elect sign, Disjoint merge for sparse merging.
    
    Shapes:
        deltas: list of (param_shape) task vectors
        Output: (param_shape) merged
    """
    
    def __init__(self, trim_ratio: float = 0.2):
        super(Model, self).__init__()
        self.trim_ratio = trim_ratio
    
    def forward(self, *deltas: torch.Tensor) -> torch.Tensor:
        # Stack deltas
        stacked = torch.stack(deltas)  # (num_models, ...)
        
        # Trim: zero out low-magnitude values
        for i in range(len(deltas)):
            threshold = stacked[i].abs().quantile(self.trim_ratio)
            stacked[i] = torch.where(stacked[i].abs() >= threshold, stacked[i], torch.zeros_like(stacked[i]))
        
        # Elect sign: majority vote
        signs = stacked.sign()
        elected_sign = signs.sum(dim=0).sign()
        
        # Disjoint merge: average values with matching sign
        mask = (signs == elected_sign.unsqueeze(0)) | (stacked == 0)
        masked = stacked * mask
        
        # Average non-zero values
        counts = (masked != 0).sum(dim=0).clamp(min=1)
        merged = masked.sum(dim=0) / counts
        
        return merged


param_shape = (4096, 4096)

def get_inputs():
    d1 = torch.randn(*param_shape, device='cuda') * 0.1
    d2 = torch.randn(*param_shape, device='cuda') * 0.1
    d3 = torch.randn(*param_shape, device='cuda') * 0.1
    return [d1, d2, d3]

def get_init_inputs():
    return [0.2]

