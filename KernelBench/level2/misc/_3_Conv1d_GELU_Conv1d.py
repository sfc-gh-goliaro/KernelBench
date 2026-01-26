import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Conv1d + GELU + Conv1d
    
    Used by: Whisper, audio encoder frontend
    
    Strided Conv1d + GELU + strided Conv1d.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.gelu(self.conv1(x)))


batch_size, in_channels, hidden_channels, seq_len = 8, 80, 1024, 3000

def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len, device='cuda')]

def get_init_inputs():
    return [in_channels, hidden_channels]

