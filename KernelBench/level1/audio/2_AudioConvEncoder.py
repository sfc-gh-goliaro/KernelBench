import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Audio Convolutional Encoder
    
    Used by: Whisper audio encoder
    
    Strided Conv1d stack (typically 2 layers) converting mel features
    to encoder input sequence.
    
    Shapes:
        Input: (batch, n_mels, time_frames)
        Output: (batch, seq_len, hidden_size)
    """
    
    def __init__(self, n_mels: int = 80, hidden_size: int = 1024):
        """
        Initialize audio conv encoder.
        
        Args:
            n_mels: Number of mel bands
            hidden_size: Output hidden dimension
        """
        super(Model, self).__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        
        # Two conv layers as in Whisper
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram (batch, n_mels, time_frames)
            
        Returns:
            Encoder features (batch, seq_len, hidden_size)
        """
        # First conv + GELU
        x = self.conv1(mel_spec)
        x = F.gelu(x)
        
        # Second conv + GELU (with stride=2 downsampling)
        x = self.conv2(x)
        x = F.gelu(x)
        
        # Transpose to (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 8, "n_mels": 80, "time_frames": 3000, "hidden_size": 1024},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("audio", "2_AudioConvEncoder")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    mel_spec = DISTRIBUTIONS[dist_name]((p["batch_size"], p["n_mels"], p["time_frames"]), dtype=dtype, device=device)
    return [mel_spec]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["n_mels"], p["hidden_size"]]
