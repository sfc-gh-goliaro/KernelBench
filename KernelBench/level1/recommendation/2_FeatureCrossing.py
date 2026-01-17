import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Feature Crossing
    
    Used by: DCN, DeepFM
    
    Explicit feature crossing via outer product or Hadamard.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, interaction_dim)
    """
    
    def __init__(self, num_features: int, embed_dim: int):
        super(Model, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        # Pairwise interactions
        self.num_interactions = num_features * (num_features - 1) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        interactions = []
        
        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                interactions.append((x[:, i] * x[:, j]).sum(dim=-1, keepdim=True))
        
        return torch.cat(interactions, dim=-1)


batch_size = 4096
num_features = 26
embed_dim = 16

def get_inputs():
    x = torch.randn(batch_size, num_features, embed_dim, device='cuda')
    return [x]

def get_init_inputs():
    return [num_features, embed_dim]

