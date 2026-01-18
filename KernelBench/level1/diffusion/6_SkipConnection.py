import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Skip Connection (Diffusion Models)

    Used by: UNet, Stable Diffusion, DiT

    Skip connections in UNet architecture that concatenate encoder features
    with decoder features at corresponding resolutions. May include
    channel projection when dimensions don't match.

    Shapes:
        encoder_features: (batch, enc_channels, height, width)
        decoder_features: (batch, dec_channels, height, width)
        Output: (batch, out_channels, height, width)
    """

    def __init__(self, encoder_channels: int, decoder_channels: int,
                 out_channels: int = None, mode: str = 'concat'):
        """
        Initialize skip connection.

        Args:
            encoder_channels: Number of encoder feature channels
            decoder_channels: Number of decoder feature channels
            out_channels: Output channels (default: decoder_channels)
            mode: 'concat' or 'add' for combining features
        """
        super(Model, self).__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels or decoder_channels
        self.mode = mode

        if mode == 'concat':
            # Project concatenated features
            self.proj = nn.Sequential(
                nn.Conv2d(encoder_channels + decoder_channels, self.out_channels,
                         kernel_size=1, bias=False),
                nn.GroupNorm(32, self.out_channels),
                nn.SiLU()
            )
        else:  # add mode
            # Project encoder to match decoder if needed
            if encoder_channels != decoder_channels:
                self.encoder_proj = nn.Conv2d(encoder_channels, decoder_channels,
                                             kernel_size=1, bias=False)
            else:
                self.encoder_proj = nn.Identity()

            if decoder_channels != self.out_channels:
                self.out_proj = nn.Conv2d(decoder_channels, self.out_channels,
                                         kernel_size=1, bias=False)
            else:
                self.out_proj = nn.Identity()

    def forward(self, encoder_features: torch.Tensor,
                decoder_features: torch.Tensor) -> torch.Tensor:
        """
        Combine encoder and decoder features.

        Args:
            encoder_features: Features from encoder (batch, enc_ch, h, w)
            decoder_features: Features from decoder (batch, dec_ch, h, w)

        Returns:
            Combined features (batch, out_channels, h, w)
        """
        if self.mode == 'concat':
            # Concatenate along channel dimension
            combined = torch.cat([encoder_features, decoder_features], dim=1)
            return self.proj(combined)
        else:
            # Add with projection
            projected_encoder = self.encoder_proj(encoder_features)
            combined = projected_encoder + decoder_features
            return self.out_proj(combined)


# ============================================================================
# Benchmark Configuration
# ============================================================================


PARAMETERS = [
    {"batch_size": 4, "encoder_channels": 512, "decoder_channels": 512, "out_channels": 512, "height": 32, "width": 32},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("diffusion", "6_SkipConnection")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    encoder_features = DISTRIBUTIONS[dist_name]((p["batch_size"], p["encoder_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    decoder_features = DISTRIBUTIONS[dist_name]((p["batch_size"], p["decoder_channels"], p["height"], p["width"]), dtype=dtype, device=device)
    return [encoder_features, decoder_features]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    p = PARAMETERS[param_idx]
    return [p["encoder_channels"], p["decoder_channels"], p["out_channels"]]
