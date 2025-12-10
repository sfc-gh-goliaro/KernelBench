import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Sparse Feed-Forward Network with Activation Sparsity.
    
    Exploits sparsity in ReLU activations to reduce computation
    by only computing active neurons.
    
    Based on: "Deja Vu" and similar activation-sparse approaches
    """
    def __init__(self, dim, hidden_dim, sparsity_threshold=0.0, 
                 use_gelu=False, top_k=None):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Hidden layer dimension
        :param sparsity_threshold: Threshold for zeroing activations
        :param use_gelu: Use GELU (approximated sparse) vs ReLU
        :param top_k: If set, keep only top-k activations
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.sparsity_threshold = sparsity_threshold
        self.use_gelu = use_gelu
        self.top_k = top_k
        
        # Standard FFN layers
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Predictor for activation sparsity (predicts which neurons will be active)
        self.predictor = nn.Linear(dim, hidden_dim)
        
        # Statistics tracking
        self.register_buffer('activation_counts', torch.zeros(hidden_dim))
        self.register_buffer('total_count', torch.zeros(1))
    
    def predict_active_neurons(self, x):
        """
        Predict which neurons will be active to skip computation.
        """
        # Use a lightweight predictor
        pred_logits = self.predictor(x)
        
        if self.top_k is not None:
            # Keep top-k neurons based on prediction
            _, top_indices = torch.topk(pred_logits.abs(), self.top_k, dim=-1)
            mask = torch.zeros_like(pred_logits)
            mask.scatter_(-1, top_indices, 1.0)
        else:
            # Use threshold
            mask = (pred_logits.abs() > self.sparsity_threshold).float()
        
        return mask
    
    def sparse_activation(self, x, mask=None):
        """
        Apply activation with sparsity.
        """
        if self.use_gelu:
            activated = F.gelu(x)
        else:
            activated = F.relu(x)
        
        # Apply sparsity mask if provided
        if mask is not None:
            activated = activated * mask
        
        # Apply threshold
        if self.sparsity_threshold > 0:
            activated = torch.where(
                activated.abs() > self.sparsity_threshold,
                activated,
                torch.zeros_like(activated)
            )
        
        return activated
    
    def forward(self, x, use_predictor=False):
        """
        Forward pass with sparse activations.
        
        :param x: Input tensor (batch, seq_len, dim)
        :param use_predictor: Whether to use the predictor for sparsity
        :return: Output tensor (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        
        # First linear layer
        hidden = self.fc1(x)
        
        # Predict active neurons (optional)
        mask = None
        if use_predictor:
            mask = self.predict_active_neurons(x)
        
        # Sparse activation
        hidden = self.sparse_activation(hidden, mask)
        
        # Track activation statistics
        if self.training:
            with torch.no_grad():
                active = (hidden.abs() > 0).float().sum(dim=(0, 1))
                self.activation_counts += active
                self.total_count += hidden.shape[0] * hidden.shape[1]
        
        # Second linear layer
        output = self.fc2(hidden)
        
        return output + residual
    
    def get_sparsity_stats(self):
        """Get activation sparsity statistics."""
        if self.total_count.item() == 0:
            return 0.0
        
        avg_active = self.activation_counts / self.total_count
        sparsity = 1.0 - avg_active.mean().item()
        return sparsity


# Test parameters
batch_size = 32
seq_len = 512
dim = 4096
hidden_dim = 11008  # Typical LLaMA FFN ratio

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim]

