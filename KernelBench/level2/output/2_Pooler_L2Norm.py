import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Pooler + L2 Normalization
    
    Used by: Embedding models (E5, GTE, BGE)
    
    CLS token extraction or mean pooling + L2 normalization.
    """
    
    def __init__(self, pool_type: str = 'cls'):
        super(Model, self).__init__()
        self.pool_type = pool_type
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'cls':
            pooled = hidden_states[:, 0]
        else:
            pooled = hidden_states.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)


batch_size, seq_len, hidden_size = 32, 512, 768

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda')]

def get_init_inputs():
    return ['cls']

