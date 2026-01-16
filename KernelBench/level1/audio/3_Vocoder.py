import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Vocoder (HiFi-GAN style)
    
    Used by: Qwen2.5-Omni (talker)
    
    HiFi-GAN style neural vocoder: upsampling convolutions +
    residual blocks for waveform synthesis from mel spectrogram.
    
    Shapes:
        Input: (batch, n_mels, time_frames)
        Output: (batch, 1, samples) audio waveform
    """
    
    def __init__(self, n_mels: int = 80, upsample_rates: list = [8, 8, 2, 2],
                 hidden_channels: int = 512):
        """
        Initialize vocoder.
        
        Args:
            n_mels: Number of mel bands
            upsample_rates: Upsampling rates per layer
            hidden_channels: Hidden channel dimension
        """
        super(Model, self).__init__()
        self.n_mels = n_mels
        self.upsample_rates = upsample_rates
        
        # Initial conv
        self.conv_pre = nn.Conv1d(n_mels, hidden_channels, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = hidden_channels
        for rate in upsample_rates:
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, rate * 2, stride=rate, padding=rate // 2)
            )
            ch = ch // 2
        
        # Final conv
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel spectrogram.
        
        Args:
            mel_spec: Mel spectrogram (batch, n_mels, time_frames)
            
        Returns:
            Audio waveform (batch, 1, samples)
        """
        x = self.conv_pre(mel_spec)
        
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================

PARAMETERS = [
    {"batch_size": 8, "n_mels": 80, "time_frames": 300},  # ~3 seconds at 100 frames/sec
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("audio", "3_Vocoder")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    shape = (p["batch_size"], p["n_mels"], p["time_frames"])
    mel_spec = DISTRIBUTIONS[dist_name](shape, dtype=dtype, device=device)
    return [mel_spec]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["n_mels"]]
