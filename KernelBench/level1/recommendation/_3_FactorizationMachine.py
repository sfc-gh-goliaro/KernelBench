import os
import sys
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Factorization Machine
    
    Used by: DeepFM, FFM
    
    FM: linear + sum of pairwise feature interactions.
    Efficient O(n*k) computation via sum-of-squares trick.
    
    Shapes:
        Input: (batch, num_features, embed_dim)
        Output: (batch, 1)
    """
    
    def __init__(self, num_features: int, embed_dim: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(num_features * embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Sum of squares trick for pairwise interactions
        sum_squared = x.sum(dim=1).pow(2).sum(dim=1, keepdim=True)
        squared_sum = x.pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        interactions = 0.5 * (sum_squared - squared_sum)
        
        # Linear term
        linear = self.linear(x.view(batch_size, -1))
        
        return linear + interactions
