"""
VQ-VAE-2 (Hierarchical Vector Quantized VAE)

A hierarchical generative model implementing VQ-VAE-2:
- Multi-scale encoder/decoder
- Hierarchical vector quantization (top and bottom levels)
- Codebook learning with EMA updates
- Commitment loss for stable training

Reference: Generating Diverse High-Fidelity Images with VQ-VAE-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with 3x3 convolutions."""
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = None):
        super().__init__()
        hidden_channels = hidden_channels or out_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + self.skip(x))


class ResidualStack(nn.Module):
    """Stack of residual blocks."""
    def __init__(self, in_channels: int, hidden_channels: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(in_channels, in_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with EMA codebook updates.
    
    Maps continuous latents to discrete codebook entries.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        # EMA update buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)

        # Distances to codebook entries
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))

        # Find nearest codebook entries
        encoding_indices = d.argmin(dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # EMA codebook update (training only)
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            dw = encodings.t() @ z_flat
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Back to (B, C, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

        return z_q, commitment_loss, encoding_indices


class EncoderBottom(nn.Module):
    """Bottom-level encoder (higher resolution)."""
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, num_res_blocks: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.res_stack = ResidualStack(hidden_channels, hidden_channels, num_res_blocks)
        self.conv_out = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.res_stack(h)
        return self.conv_out(h)


class EncoderTop(nn.Module):
    """Top-level encoder (lower resolution)."""
    def __init__(self, embedding_dim: int, hidden_channels: int, num_res_blocks: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_dim, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1)
        self.res_stack = ResidualStack(hidden_channels, hidden_channels, num_res_blocks)
        self.conv_out = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)

    def forward(self, z_bottom: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(z_bottom))
        h = F.relu(self.conv2(h))
        h = self.res_stack(h)
        return self.conv_out(h)


class DecoderTop(nn.Module):
    """Top-level decoder."""
    def __init__(self, embedding_dim: int, hidden_channels: int, num_res_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)
        self.res_stack = ResidualStack(hidden_channels, hidden_channels, num_res_blocks)
        self.upsample = nn.ConvTranspose2d(hidden_channels, embedding_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, z_q_top: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv_in(z_q_top))
        h = self.res_stack(h)
        return self.upsample(h)


class DecoderBottom(nn.Module):
    """Bottom-level decoder."""
    def __init__(self, embedding_dim: int, hidden_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()
        # Input: concatenation of z_q_bottom and decoded z_q_top
        self.conv_in = nn.Conv2d(embedding_dim * 2, hidden_channels, kernel_size=3, padding=1)
        self.res_stack = ResidualStack(hidden_channels, hidden_channels, num_res_blocks)
        self.upsample1 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z_q_bottom: torch.Tensor, dec_top: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_q_bottom, dec_top], dim=1)
        h = F.relu(self.conv_in(h))
        h = self.res_stack(h)
        h = F.relu(self.upsample1(h))
        return torch.tanh(self.upsample2(h))


class Model(nn.Module):
    """VQ-VAE-2 Hierarchical Generative Model."""
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        num_res_blocks: int = 2,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        # Bottom-level
        self.encoder_bottom = EncoderBottom(
            in_channels, hidden_channels, embedding_dim, num_res_blocks
        )
        self.vq_bottom = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

        # Top-level
        self.encoder_top = EncoderTop(
            embedding_dim, hidden_channels, num_res_blocks
        )
        self.vq_top = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

        # Decoders
        self.decoder_top = DecoderTop(
            embedding_dim, hidden_channels, num_res_blocks
        )
        self.decoder_bottom = DecoderBottom(
            embedding_dim, hidden_channels, in_channels, num_res_blocks
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bottom encoder
        z_bottom = self.encoder_bottom(x)

        # Top encoder
        z_top = self.encoder_top(z_bottom)

        return z_bottom, z_top

    def decode(self, z_q_bottom: torch.Tensor, z_q_top: torch.Tensor) -> torch.Tensor:
        # Decode top
        dec_top = self.decoder_top(z_q_top)

        # Decode bottom (with top context)
        x_recon = self.decoder_bottom(z_q_bottom, dec_top)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        z_bottom, z_top = self.encode(x)

        # Quantize
        z_q_top, loss_top, indices_top = self.vq_top(z_top)
        z_q_bottom, loss_bottom, indices_bottom = self.vq_bottom(z_bottom)

        # Decode
        x_recon = self.decode(z_q_bottom, z_q_top)

        # Total commitment loss
        commitment_loss = loss_top + loss_bottom

        return x_recon, commitment_loss, indices_top, indices_bottom


# Configuration
batch_size = 8
img_size = 256
in_channels = 3
hidden_channels = 128
embedding_dim = 64
num_embeddings = 512
num_res_blocks = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, img_size, img_size)]


def get_init_inputs():
    return [{
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'embedding_dim': embedding_dim,
        'num_embeddings': num_embeddings,
        'num_res_blocks': num_res_blocks,
    }]

