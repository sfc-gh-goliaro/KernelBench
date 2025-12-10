import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Mel Spectrogram computation for audio processing.
    
    Converts raw audio waveform to mel-frequency spectrogram
    representation used in speech models like Whisper.
    
    Based on: Whisper and standard audio processing pipelines
    """
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, 
                 n_mels=80, f_min=0, f_max=8000):
        """
        :param sample_rate: Audio sample rate in Hz
        :param n_fft: FFT window size
        :param hop_length: Hop length between frames
        :param n_mels: Number of mel filterbank channels
        :param f_min: Minimum frequency for mel filterbank
        :param f_max: Maximum frequency for mel filterbank
        """
        super(Model, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Create mel filterbank
        mel_filters = self._create_mel_filterbank(
            n_fft, n_mels, sample_rate, f_min, f_max
        )
        self.register_buffer('mel_filters', mel_filters)
        
        # Create Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
    
    def _hz_to_mel(self, freq):
        """Convert Hz to mel scale."""
        return 2595 * math.log10(1 + freq / 700)
    
    def _mel_to_hz(self, mel):
        """Convert mel to Hz scale."""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _create_mel_filterbank(self, n_fft, n_mels, sample_rate, f_min, f_max):
        """Create mel filterbank matrix."""
        # Frequency bins
        n_freqs = n_fft // 2 + 1
        freqs = torch.linspace(0, sample_rate / 2, n_freqs)
        
        # Mel scale points
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m) for m in mel_points])
        
        # Create filterbank
        filterbank = torch.zeros(n_mels, n_freqs)
        
        for i in range(n_mels):
            # Lower and upper slopes
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]
            
            # Rising slope
            rising = (freqs - lower) / (center - lower + 1e-10)
            # Falling slope
            falling = (upper - freqs) / (upper - center + 1e-10)
            
            filterbank[i] = torch.maximum(
                torch.zeros_like(freqs),
                torch.minimum(rising, falling)
            )
        
        return filterbank
    
    def forward(self, audio):
        """
        Compute mel spectrogram from audio waveform.
        
        :param audio: Audio waveform (batch, samples) or (batch, 1, samples)
        :return: Mel spectrogram (batch, n_mels, time_frames)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        batch_size = audio.shape[0]
        
        # Pad audio for STFT
        pad_amount = self.n_fft // 2
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')
        
        # Compute STFT using unfold
        # Frame the signal
        frames = audio.unfold(-1, self.n_fft, self.hop_length)
        
        # Apply window
        frames = frames * self.window.unsqueeze(0).unsqueeze(0)
        
        # FFT
        spectrum = torch.fft.rfft(frames, dim=-1)
        
        # Magnitude
        magnitude = spectrum.abs()
        
        # Apply mel filterbank
        mel_spec = torch.matmul(magnitude, self.mel_filters.t())
        
        # Transpose to (batch, n_mels, time)
        mel_spec = mel_spec.transpose(-1, -2)
        
        # Log scale (with small epsilon for stability)
        log_mel_spec = torch.log(mel_spec.clamp(min=1e-10))
        
        # Normalize (Whisper-style)
        log_mel_spec = torch.clamp(log_mel_spec, min=log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0
        
        return log_mel_spec


# Test parameters
batch_size = 8
audio_length = 480000  # 30 seconds at 16kHz
sample_rate = 16000
n_fft = 400
hop_length = 160
n_mels = 80

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    audio = torch.randn(batch_size, audio_length)
    return [audio]

def get_init_inputs():
    return [sample_rate, n_fft, hop_length, n_mels]

