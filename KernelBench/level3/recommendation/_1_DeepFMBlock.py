import torch
import torch.nn as nn
import torch.nn.functional as F

class FMLayer(nn.Module):
    """Factorization Machine layer for second-order feature interactions."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute FM second-order interactions.
        
        Args:
            x: Embedding tensor (batch, num_fields, embedding_dim)
        
        Returns:
            FM output (batch, 1)
        """
        # Sum of squares
        sum_of_square = torch.sum(x, dim=1).pow(2)  # (batch, embedding_dim)
        # Square of sums
        square_of_sum = torch.sum(x.pow(2), dim=1)  # (batch, embedding_dim)
        # FM cross term: 0.5 * sum((sum_i x_i)^2 - sum_i x_i^2)
        cross_term = 0.5 * (sum_of_square - square_of_sum)  # (batch, embedding_dim)
        return cross_term.sum(dim=1, keepdim=True)  # (batch, 1)


class DeepMLP(nn.Module):
    """Deep component MLP."""
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Model(nn.Module):
    """
    DeepFM Block
    
    The core architecture block for DeepFM recommendation model.
    Used by: DeepFM for click-through rate prediction
    
    Architecture:
        Input (sparse + dense features)
          -> Embedding lookup
          -> FM Layer (second-order interactions) -> FM output
          -> Deep MLP (high-order interactions) -> Deep output
        FM output + Deep output -> Final prediction
    
    Key features:
    - Factorization Machine for automatic feature interactions
    - Deep MLP for high-order feature combinations
    - Joint training of FM and Deep components
    """
    def __init__(self, num_fields: int, embedding_dim: int, hidden_dims: list = [256, 128, 64],
                 num_embeddings: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # Sparse feature embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
        # First-order (linear) weights
        self.linear = nn.Embedding(num_embeddings, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # FM component
        self.fm = FMLayer()
        
        # Deep component
        deep_input_dim = num_fields * embedding_dim
        self.deep = DeepMLP(deep_input_dim, hidden_dims, dropout)
        
        # Final prediction layers
        self.fm_final = nn.Linear(1, 1, bias=False)
        self.deep_final = nn.Linear(hidden_dims[-1], 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse feature indices (batch, num_fields)
        
        Returns:
            Click probability logits (batch, 1)
        """
        # Embedding lookup
        embed = self.embeddings(x)  # (batch, num_fields, embedding_dim)
        
        # First-order (linear) component
        linear_out = self.linear(x).sum(dim=1) + self.bias  # (batch, 1)
        
        # FM second-order component
        fm_out = self.fm(embed)  # (batch, 1)
        
        # Deep component
        deep_input = embed.view(embed.size(0), -1)  # (batch, num_fields * embedding_dim)
        deep_out = self.deep(deep_input)  # (batch, hidden_dims[-1])
        deep_out = self.deep_final(deep_out)  # (batch, 1)
        
        # Combine all components
        output = linear_out + fm_out + deep_out
        
        return output


# Benchmark configuration
batch_size = 1024
num_fields = 39  # Criteo dataset has 39 features
embedding_dim = 16
num_embeddings = 10000

def get_inputs():
    # Simulated sparse feature indices
    x = torch.randint(0, num_embeddings, (batch_size, num_fields))
    return [x]

def get_init_inputs():
    return [num_fields, embedding_dim]

