import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Vocoder Decoder (TTS/Speech Synthesis)

    Used by: HiFi-GAN, BigVGAN, Vocos

    Converts mel spectrograms to raw audio waveforms. Uses transposed
    convolutions with multi-receptive field fusion for high-fidelity audio.

    Shapes:
        mel_spectrogram: (batch, n_mels, mel_frames) mel features
        Output: (batch, 1, audio_samples) raw waveform
    """

    def __init__(self, n_mels: int = 80, upsample_rates: list = None,
                 upsample_kernel_sizes: list = None, hidden_channels: int = 512):
        """
        Initialize vocoder decoder.

        Args:
            n_mels: Number of mel channels
            upsample_rates: Upsampling rates for each layer
            upsample_kernel_sizes: Kernel sizes for transposed convs
            hidden_channels: Number of hidden channels
        """
        super(Model, self).__init__()

        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]  # Total: 256x upsampling
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]

        self.n_mels = n_mels
        self.hidden_channels = hidden_channels

        # Initial conv
        self.conv_pre = nn.Conv1d(n_mels, hidden_channels, 7, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = hidden_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel, stride=rate,
                                  padding=(kernel - rate) // 2)
            )
            ch = ch // 2

        # Multi-receptive field fusion (MRF) blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(upsample_rates)):
            ch = hidden_channels // (2 ** (i + 1))
            self.resblocks.append(self._make_resblock(ch))

        # Final conv
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

    def _make_resblock(self, channels: int) -> nn.Module:
        """Create a residual block with multiple dilations."""
        return nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, 3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4),
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel spectrogram.

        Args:
            mel_spectrogram: Mel spectrogram (batch, n_mels, mel_frames)

        Returns:
            Audio waveform (batch, 1, audio_samples)
        """
        x = self.conv_pre(mel_spectrogram)

        for up, resblock in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = x + resblock(x)  # Residual connection

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 4, "n_mels": 80, "mel_frames": 200},
    # HiFi-GAN: High-fidelity generative adversarial vocoder
    {"batch_size": 8, "n_mels": 80, "mel_frames": 128},
    # BigVGAN: Large-scale universal vocoder
    {"batch_size": 2, "n_mels": 100, "mel_frames": 400},
    # Vocos: Closing the gap between time-domain and Fourier-based vocoders
    {"batch_size": 16, "n_mels": 80, "mel_frames": 64},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("tts", "6_VocoderDecoder")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    mel_spectrogram = DISTRIBUTIONS[dist_name]((p["batch_size"], p["n_mels"], p["mel_frames"]), dtype=dtype, device=device)
    return [mel_spectrogram]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["n_mels"]]
