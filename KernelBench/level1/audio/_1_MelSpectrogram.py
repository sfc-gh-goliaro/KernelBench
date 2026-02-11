import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional


def _hertz_to_mel_slaney(freq):
    """Convert Hz to Slaney mel scale (matches librosa / HuggingFace)."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels):
    """Convert Slaney mel scale to Hz (matches librosa / HuggingFace)."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq


def create_mel_filterbank_slaney(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """
    Create a Slaney-normalized mel filterbank (matches HuggingFace / librosa).

    Returns:
        np.ndarray of shape (num_frequency_bins, num_mel_filters)
    """
    mel_min = _hertz_to_mel_slaney(min_frequency)
    mel_max = _hertz_to_mel_slaney(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)

    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    # Triangular filters
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    # Slaney area normalization
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)

    return mel_filters


class Model(nn.Module):
    """
    Mel Spectrogram
    
    Used by: Whisper, Qwen2-Audio, Ultravox
    
    Short-time Fourier transform + mel filterbank to convert
    waveform to mel spectrogram.
    
    Shapes:
        Input: (batch, samples) audio waveform
        Output: (batch, n_mels, time_frames) mel spectrogram

    Supports two mel filterbank modes:
        - mel_scale="htk" (default): Standard HTK mel scale
        - mel_scale="slaney": Slaney/librosa mel scale with area normalization
          (used by Whisper, HuggingFace WhisperFeatureExtractor)

    Supports two log modes:
        - log_mel="log" (default): Natural log with epsilon floor
        - log_mel="log10": log10 with clamp, used by Whisper
    """
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160,
                 n_mels: int = 80, f_min: float = 0.0, f_max: float = None,
                 mel_scale: str = "htk", log_mel: str = "log",
                 mel_filters_np: Optional[np.ndarray] = None):
        """
        Initialize mel spectrogram.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sample_rate / 2)
            mel_scale: "htk" or "slaney" mel frequency scale
            log_mel: "log" for natural log, "log10" for base-10 log
            mel_filters_np: Pre-computed mel filterbank (num_freq_bins, n_mels).
                            If provided, overrides mel_scale/f_min/f_max.
        """
        super(Model, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.mel_scale = mel_scale
        self.log_mel = log_mel
        
        # Create mel filterbank
        if mel_filters_np is not None:
            # Use pre-computed filters (num_freq_bins, n_mels) -> transpose to (n_mels, num_freq_bins)
            mel_fb = torch.from_numpy(mel_filters_np.T).float()
        elif mel_scale == "slaney":
            num_freq_bins = n_fft // 2 + 1
            filters_np = create_mel_filterbank_slaney(
                num_freq_bins, n_mels, f_min, self.f_max, sample_rate,
            )
            # filters_np is (num_freq_bins, n_mels), transpose to (n_mels, num_freq_bins)
            mel_fb = torch.from_numpy(filters_np.T).float()
        else:
            mel_fb = self._create_mel_filterbank_htk()
        # Register mel filterbank and Hann window as buffers so they follow
        # .to(device=...) calls.  We also keep numpy copies so forward() can
        # always reconstruct full-precision float32 tensors even after the
        # parent model is cast to a lower dtype (e.g. bfloat16).
        self.register_buffer('mel_fb', mel_fb)
        self._mel_fb_np = mel_fb.numpy().copy()
        
        # Create Hann window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
        self._window_np = window.numpy().copy()
    
    def _hz_to_mel(self, freq: float) -> float:
        """Convert Hz to HTK mel scale."""
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert HTK mel scale to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank_htk(self) -> torch.Tensor:
        """Create HTK-style mel filterbank matrix (n_mels, num_freq_bins)."""
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
            waveform: Audio waveform (batch, samples), should be float32
            
        Returns:
            Mel spectrogram (batch, n_mels, time_frames)
        """
        # STFT requires float32.  We recreate the Hann window directly on the
        # target device (torch.hann_window uses device-native math, so GPU and
        # CPU windows can differ by ~6e-8; creating on-device matches the HF
        # WhisperFeatureExtractor which does the same).
        # The mel filterbank is reconstructed from a numpy copy to guarantee
        # full float32 precision even when the parent model has been cast to a
        # lower dtype (e.g. bfloat16 degrades the registered buffer copy).
        waveform = waveform.float()
        window = torch.hann_window(self.n_fft, device=waveform.device)
        mel_fb = torch.from_numpy(self._mel_fb_np).to(device=waveform.device)

        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True
        )
        
        # Power spectrogram
        power_spec = stft.abs().pow(2)  # (batch, freq, time)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(mel_fb, power_spec)  # (batch, n_mels, time)
        
        if self.log_mel == "log10":
            # Whisper-style: log10 with clamp floor
            log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        else:
            # Default: natural log with epsilon
            log_mel = torch.log(mel_spec + 1e-10)
        
        return log_mel
