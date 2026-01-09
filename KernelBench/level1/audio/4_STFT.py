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

batch_size = 16
num_samples = 16000 * 30  # 30 seconds at 16kHz
n_fft = 400
hop_length = 160
win_length = 400
window = 'hann'

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    waveform = torch.randn(batch_size, num_samples, device='cuda')
    return [waveform]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [n_fft, hop_length, win_length, window]

