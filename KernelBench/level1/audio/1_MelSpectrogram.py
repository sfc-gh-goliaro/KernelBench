import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Mel Spectrogram
    
    Used by: Whisper, Qwen2-Audio, Ultravox
    
    Short-time Fourier transform + mel filterbank to convert
    waveform to mel spectrogram.
    
    Shapes:
        Input: (batch, samples) audio waveform
        Output: (batch, n_mels, time_frames) mel spectrogram
    """
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160,
                 n_mels: int = 80, f_min: float = 0.0, f_max: float = None):
        """
        Initialize mel spectrogram.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sample_rate / 2)
        """
        super(Model, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
        # Create mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_fb', mel_fb)
        
        # Create Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
    
    def _hz_to_mel(self, freq: float) -> float:
        """Convert Hz to mel scale."""
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel scale to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Mel points
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m) for m in mel_points])
        
        # FFT bin frequencies
        fft_bins = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        
        # Create filterbank
        filterbank = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        
        for i in range(self.n_mels):
            f_left = hz_points[i]
            f_center = hz_points[i + 1]
            f_right = hz_points[i + 2]
            
            # Left slope
            left_mask = (fft_bins >= f_left) & (fft_bins <= f_center)
            filterbank[i, left_mask] = (fft_bins[left_mask] - f_left) / (f_center - f_left + 1e-10)
            
            # Right slope
            right_mask = (fft_bins >= f_center) & (fft_bins <= f_right)
            filterbank[i, right_mask] = (f_right - fft_bins[right_mask]) / (f_right - f_center + 1e-10)
        
        return filterbank
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.
        
        Args:
            waveform: Audio waveform (batch, samples)
            
        Returns:
            Mel spectrogram (batch, n_mels, time_frames)
        """
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True
        )
        
        # Power spectrogram
        power_spec = stft.abs().pow(2)  # (batch, freq, time)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_fb, power_spec)  # (batch, n_mels, time)
        
        # Log mel spectrogram
        log_mel = torch.log(mel_spec + 1e-10)
        
        return log_mel


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
sample_rate = 16000
duration = 30  # seconds
n_samples = sample_rate * duration
n_mels = 80

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    waveform = torch.randn(batch_size, n_samples, device='cuda')
    return [waveform]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [sample_rate, 400, 160, n_mels]

