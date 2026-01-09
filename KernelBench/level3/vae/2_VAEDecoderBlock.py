import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for VAE decoder."""
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
    VAE Decoder Block
    
    The core upsampling block in VAE decoders.
    Used by: VAE, VQ-VAE, SD VAE (Stable Diffusion)
    
    Architecture:
        x -> ResidualBlock
          -> ResidualBlock
          -> Upsample (2x) -> Conv -> BN -> ReLU
        Output: Features at double spatial resolution
    
    This block is repeated multiple times to progressively
    upsample from latent space to output image dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(in_channels if i == 0 else in_channels, in_channels)
            for i in range(num_res_blocks)
        ])
        
        # Upsampling (bilinear + conv is smoother than transposed conv)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


# Benchmark configuration
batch_size = 8
in_channels = 128
out_channels = 64
height = 32
width = 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels]

