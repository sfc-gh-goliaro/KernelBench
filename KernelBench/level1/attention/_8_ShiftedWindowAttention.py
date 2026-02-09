import os
import sys
import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from ..matmul._10_Linear import Model as Linear
from ..matmul._1_MatMul import Model as MatMul
from ..activations._1_ReLU import Model as ReLU
from ..activations._5_Softmax import Model as Softmax
from ..activations._3_Sigmoid import Model as Sigmoid


class Model(nn.Module):
    """
    Shifted Window Attention (SwinV2)

    Used by: Swin Transformer V2

    Window-based self-attention with cosine similarity scoring and
    continuous log-spaced relative position bias via MLP.

    NOTE: Q/K/V projections are done separately using Linear operators.
    This operator takes pre-projected Q, K, V tensors (already in
    multi-head format).

    Key SwinV2-specific features:
    - Cosine attention: normalize(Q) @ normalize(K).T * exp(logit_scale)
    - Learnable per-head logit_scale parameter (clamped)
    - Continuous position bias via MLP: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
      applied to log-spaced relative coordinate table, then scaled by 16 * sigmoid
    - Attention mask applied twice (matches HuggingFace implementation)

    Shapes:
        q:      (num_windows * batch, num_heads, window_size^2, head_dim)
        k:      (num_windows * batch, num_heads, window_size^2, head_dim)
        v:      (num_windows * batch, num_heads, window_size^2, head_dim)
        Output: (num_windows * batch, window_size^2, dim)

    Uses level1 operators:
    - Linear from level1/matmul/10_Linear (cpb_mlp)
    - MatMul from level1/matmul/1_MatMul (attention scores & context)
    - ReLU from level1/activations/1_ReLU (cpb_mlp)
    - Softmax from level1/activations/5_Softmax (attention weights)
    - Sigmoid from level1/activations/3_Sigmoid (position bias scaling)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        pretrained_window_size: int = 0,
    ):
        """
        Initialize SwinV2 shifted window attention.

        Args:
            dim: Input/output dimension (num_heads * head_dim)
            num_heads: Number of attention heads
            window_size: Size of attention window (scalar; used as (W, W))
            pretrained_window_size: Window size used during pre-training
                                    (for normalizing the coordinate table)
        """
        super(Model, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.all_head_size = num_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )

        # ---- Learnable logit scale for cosine attention ----
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1)))
        )

        # ---- Continuous relative position bias MLP ----
        # HF structure: nn.Sequential(Linear(2,512,bias), ReLU, Linear(512,heads,no bias))
        # Weight keys: continuous_position_bias_mlp.{0,2}.{weight,bias}
        self.continuous_position_bias_mlp = nn.Sequential(
            Linear(2, 512, bias=True),
            ReLU(),
            Linear(512, num_heads, bias=False),
        )

        # ---- Build relative coords table (matches HF exactly) ----
        pretrained_window_size = (
            pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size)
        )

        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.int64
        ).float()
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.int64
        ).float()
        relative_coords_table = (
            torch.stack(
                torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
            )
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # [1, 2*Wh-1, 2*Ww-1, 2]

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        elif window_size > 1:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / math.log2(8)
        )
        self.register_buffer(
            "relative_coords_table", relative_coords_table, persistent=False
        )

        # ---- Build relative position index (matches HF exactly) ----
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

        # ---- Level1 operators ----
        self.matmul = MatMul()
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        SwinV2 shifted window attention forward pass.

        Args:
            query: (num_windows * batch, num_heads, window_size^2, head_dim)
                Pre-projected query in multi-head format.
            key: (num_windows * batch, num_heads, window_size^2, head_dim)
                Pre-projected key in multi-head format.
            value: (num_windows * batch, num_heads, window_size^2, head_dim)
                Pre-projected value in multi-head format.
            attention_mask: Optional mask for shifted windows
                (num_windows, window_size^2, window_size^2)

        Returns:
            Output tensor (num_windows * batch, window_size^2, dim)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[2]

        # ---- Cosine attention ----
        attention_scores = self.matmul(
            F.normalize(query, dim=-1),
            F.normalize(key, dim=-1).transpose(-2, -1),
        )
        logit_scale = torch.clamp(
            self.logit_scale, max=math.log(1.0 / 0.01)
        ).exp()
        attention_scores = attention_scores * logit_scale

        # ---- Continuous relative position bias ----
        relative_position_bias_table = self.continuous_position_bias_mlp(
            self.relative_coords_table
        ).view(-1, self.num_attention_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        relative_position_bias = 16 * self.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # ---- Attention mask for shifted windows ----
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_scores = (
                attention_scores.view(
                    batch_size // mask_shape,
                    mask_shape,
                    self.num_attention_heads,
                    seq_len,
                    seq_len,
                )
                + attention_mask.unsqueeze(1).unsqueeze(0)
            )
            # HF applies mask twice (this matches their code exactly)
            attention_scores = attention_scores + attention_mask.unsqueeze(
                1
            ).unsqueeze(0)
            attention_scores = attention_scores.view(
                -1, self.num_attention_heads, seq_len, seq_len
            )

        attention_probs = self.softmax(attention_scores)

        # ---- Context computation ----
        context_layer = self.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


# ============================================================================
# Benchmark Configuration
# ============================================================================
