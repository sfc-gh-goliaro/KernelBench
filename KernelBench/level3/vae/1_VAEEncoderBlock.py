import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for VAE encoder."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Model(nn.Module):
    """
    VAE Encoder Block
    
    The core downsampling block in VAE encoders.
    Used by: VAE, VQ-VAE, SD VAE (Stable Diffusion)
    
    Architecture:
        x -> Conv (downsample) -> BN -> ReLU
          -> ResidualBlock
          -> ResidualBlock
        Output: Features at half spatial resolution
    
    This block is repeated multiple times to progressively
    downsample the input to latent space dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()
        
        # Downsampling convolution (stride 2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(out_channels, out_channels) for _ in range(num_res_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


# Benchmark configuration
batch_size = 8
in_channels = 64
out_channels = 128
height = 64
width = 64

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]

