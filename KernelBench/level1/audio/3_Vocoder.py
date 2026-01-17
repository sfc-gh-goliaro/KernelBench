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

batch_size = 8
n_mels = 80
time_frames = 300  # ~3 seconds at 100 frames/sec

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    mel_spec = torch.randn(batch_size, n_mels, time_frames, device='cuda')
    return [mel_spec]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [n_mels]

