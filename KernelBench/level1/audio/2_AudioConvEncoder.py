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

batch_size = 8
n_mels = 80
time_frames = 3000  # ~30 seconds at 100 frames/sec
hidden_size = 1024

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    mel_spec = torch.randn(batch_size, n_mels, time_frames, device='cuda')
    return [mel_spec]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [n_mels, hidden_size]

