import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Shifted Window Attention
    
    Used by: Swin Transformer
    
    Shifted window attention with cyclic shift for cross-window
    connections. Enables information flow between windows.
    
    Shapes:
        Input: (batch, height, width, channels)
        Output: (batch, height, width, channels)
    """
    
    def __init__(self, dim: int, window_size: int = 7, shift_size: int = 3, num_heads: int = 8):
        """
        Initialize shifted window attention.
        
        Args:
            dim: Input dimension
            window_size: Size of attention window
            shift_size: Shift amount for cyclic shift
            num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Create relative position index
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
        self.register_buffer('relative_position_index', relative_position_index)
    
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition into windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, C)
        return windows
    
    def _window_reverse(self, windows: torch.Tensor, H: int, W: int, B: int) -> torch.Tensor:
        """Reverse window partition."""
        x = windows.view(B, H // self.window_size, W // self.window_size,
                        self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x
    
    def _create_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create attention mask for shifted windows."""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = self._window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shifted window attention forward pass.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            Output tensor (batch, height, width, channels)
        """
        B, H, W, C = x.shape
        
        # Cyclic shift
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # Partition into windows
        x_windows = self._window_partition(shifted_x)
        B_windows = x_windows.shape[0]
        
        # Create attention mask
        attn_mask = self._create_mask(H, W, x.device)
        
        # QKV projection
        qkv = self.qkv(x_windows).reshape(B_windows, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask
        num_windows = B_windows // B
        attn = attn.view(B, num_windows, self.num_heads, -1, -1)
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(B_windows, self.num_heads, -1, -1)
        
        attn = F.softmax(attn, dim=-1)
        
        x_windows = (attn @ v).transpose(1, 2).reshape(B_windows, -1, C)
        x_windows = self.proj(x_windows)
        
        # Reverse window partition
        shifted_x = self._window_reverse(x_windows, H, W, B)
        
        # Reverse cyclic shift
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "height": 224, "width": 224, "channels": 96, "window_size": 7, "shift_size": 3, "num_heads": 4},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("attention", "15_ShiftedWindowAttention")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["height"], p["width"], p["channels"])
    x = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [x]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["channels"], p["window_size"], p["shift_size"], p["num_heads"]]
