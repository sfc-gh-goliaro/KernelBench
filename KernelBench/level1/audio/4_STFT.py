import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Short-Time Fourier Transform (STFT)
    
    Used by: Whisper, audio encoders, speech recognition models
    
    Computes the STFT of audio signals to obtain time-frequency representation.
    Used as the first step in mel-spectrogram computation.
    
    Shapes:
        Input: (batch_size, num_samples) - raw audio waveform
        Output: (batch_size, num_frames, n_fft//2 + 1) - magnitude spectrogram
    """
    
    def __init__(self, n_fft: int = 400, hop_length: int = 160,
                 win_length: int = 400, window: str = 'hann'):
        """
        Initialize STFT.
        
        Args:
            n_fft: FFT size
            hop_length: Hop length between frames
            win_length: Window length
            window: Window type ('hann', 'hamming', 'blackman')
        """
        super(Model, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Create window function
        if window == 'hann':
            win = torch.hann_window(win_length)
        elif window == 'hamming':
            win = torch.hamming_window(win_length)
        elif window == 'blackman':
            win = torch.blackman_window(win_length)
        else:
            win = torch.ones(win_length)
        
        self.register_buffer('window', win)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT magnitude spectrogram.
        
        Args:
            waveform: Audio waveform (batch_size, num_samples)
            
        Returns:
            Magnitude spectrogram (batch_size, n_fft//2 + 1, num_frames)
        """
        # Compute STFT
        stft_out = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Compute magnitude
        magnitude = torch.abs(stft_out)
        
        return magnitude


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 16, "num_samples": 16000 * 30, "n_fft": 400, "hop_length": 160, "win_length": 400, "window": 'hann'},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("audio", "4_STFT")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    waveform = DISTRIBUTIONS[dist_name]((p["batch_size"], p["num_samples"]), dtype=dtype, device=device)
    return [waveform]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["n_fft"], p["hop_length"], p["win_length"], p["window"]]
