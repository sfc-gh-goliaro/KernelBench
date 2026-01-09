import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayer(nn.Module):
    """Single cross layer for explicit feature interactions."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Cross layer operation: x0 * (xl^T * w) + b + xl
        
        Args:
            x0: Original input (batch, input_dim)
            xl: Current layer input (batch, input_dim)
        
        Returns:
            Cross layer output (batch, input_dim)
        """
        # xl^T * w -> scalar per sample (batch,)
        xl_w = (xl * self.weight).sum(dim=1, keepdim=True)  # (batch, 1)
        # x0 * (xl^T * w) + b + xl
        return x0 * xl_w + self.bias + xl


class CrossNetwork(nn.Module):
    """Cross Network with multiple cross layers."""
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        return xl


class DeepNetwork(nn.Module):
    """Deep MLP component."""
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
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
    Deep & Cross Network (DCN) Block
    
    The core architecture block for DCN recommendation model.
    Used by: DCN, DCN-V2 for click-through rate prediction
    
    Architecture:
        Input features (embedded)
          -> Cross Network (explicit feature crossing)
          -> Deep Network (implicit feature interactions)
        -> Concat -> Final prediction
    
    Key features:
    - Cross Network for bounded-degree feature interactions
    - Polynomial interactions learned automatically
    - Combined with Deep Network for complex patterns
    """
    def __init__(self, num_fields: int, embedding_dim: int, num_cross_layers: int = 3,
                 hidden_dims: list = [256, 128], num_embeddings: int = 10000, 
                 dropout: float = 0.1):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # Sparse feature embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
        input_dim = num_fields * embedding_dim
        
        # Cross Network
        self.cross_network = CrossNetwork(input_dim, num_cross_layers)
        
        # Deep Network
        self.deep_network = DeepNetwork(input_dim, hidden_dims, dropout)
        
        # Final prediction
        final_dim = input_dim + hidden_dims[-1]  # Cross output + Deep output
        self.final = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse feature indices (batch, num_fields)
        
        Returns:
            Click probability logits (batch, 1)
        """
        # Embedding lookup and flatten
        embed = self.embeddings(x)  # (batch, num_fields, embedding_dim)
        embed_flat = embed.view(embed.size(0), -1)  # (batch, num_fields * embedding_dim)
        
        # Cross Network
        cross_out = self.cross_network(embed_flat)  # (batch, input_dim)
        
        # Deep Network
        deep_out = self.deep_network(embed_flat)  # (batch, hidden_dims[-1])
        
        # Combine and predict
        combined = torch.cat([cross_out, deep_out], dim=1)
        output = self.final(combined)
        
        return output


# Benchmark configuration
batch_size = 1024
num_fields = 39
embedding_dim = 16
num_cross_layers = 3
num_embeddings = 10000

def get_inputs():
    x = torch.randint(0, num_embeddings, (batch_size, num_fields))
    return [x]

def get_init_inputs():
    return [num_fields, embedding_dim, num_cross_layers]

