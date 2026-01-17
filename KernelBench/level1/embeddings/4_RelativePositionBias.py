import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Relative Position Bias
    
    Used by: Swin Transformer, T5, DeBERTa
    
    Learnable relative position bias table that provides position-dependent
    attention biases. More flexible than absolute position embeddings.
    
    Shapes:
        Input: window_size or (query_length, key_length)
        Output: (num_heads, query_length, key_length)
    """
    
    def __init__(self, num_heads: int = 8, window_size: int = 7):
        """
        Initialize Relative Position Bias.
        
        Args:
            num_heads: Number of attention heads
            window_size: Size of local attention window
        """
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Bias table size: (2*window_size-1) x (2*window_size-1) for 2D
        # Relative positions range from -(window_size-1) to +(window_size-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self) -> torch.Tensor:
        """
        Get relative position bias for attention.
        
        Returns:
            Relative position bias (num_heads, window_size^2, window_size^2)
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 64  # Number of windows
num_heads = 8
window_size = 7

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    return []

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_heads, window_size]

