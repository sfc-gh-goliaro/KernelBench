"""
Selective and Sliding Tile Attention (SSTA) for 3D Video Diffusion

Used by: HunyuanVideo 1.5 (production inference with sparse attention)

SSTA reduces the quadratic cost of attention over long video token sequences
by combining two mechanisms:

1. **STA (Sliding Tile Attention)**: Each query tile only attends to a fixed
   3D local neighborhood of key/value tiles (e.g., a 3x3x3 window of tiles).
   This is the 3D spatiotemporal analog of sliding window attention.

2. **Selective attention**: On top of the local window, SSTA dynamically
   selects additional non-local tiles via importance sampling. A lightweight
   scoring pass identifies which distant tiles are most relevant to each
   query tile, then includes those in the attention computation.

The combined mask is: SSTA = STA_local_mask OR selective_top_k_mask

Reference: Tencent HunyuanVideo-1.5 (https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)

This implementation uses pure PyTorch (dense attention with a block-sparse mask)
as the reference. The Tencent production code uses a custom CUDA kernel
(flex_block_attn_func) for the block-sparse attention computation. The whole
point of this operator is for someone to replace the attention computation
with an optimized kernel.

Input shapes (BHSD layout):
    q: (batch_size, num_heads, seq_len, head_dim)  -- seq_len = T*H*W + text_len
    k: (batch_size, num_heads, seq_len, head_dim)
    v: (batch_size, num_heads, seq_len, head_dim)
    canvas_thw: tuple (T, H, W) -- spatiotemporal grid of video tokens

Output:
    (batch_size, num_heads, seq_len, head_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from functools import lru_cache


# ============================================================================
# Tile / Untile helpers
# ============================================================================

def tile(x: torch.Tensor, canvas_thw: Tuple[int, int, int],
         tile_thw: Tuple[int, int, int]) -> torch.Tensor:
    """Rearrange tokens into tile order for block-based attention.

    Args:
        x: (B, H, S, D) where S = T * H_spatial * W_spatial
        canvas_thw: (T, H, W) spatiotemporal dimensions
        tile_thw: (tile_t, tile_h, tile_w) tile dimensions

    Returns:
        (B, H, S, D) with tokens reordered so that each contiguous block of
        tile_t * tile_h * tile_w tokens belongs to the same tile.
    """
    B, H, S, D = x.shape
    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw

    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w

    # (B, H, S, D) -> (B, H, n_t, tile_t, n_h, tile_h, n_w, tile_w, D)
    x = x.view(B, H, n_t, tile_t, n_h, tile_h, n_w, tile_w, D)
    # -> (B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
    # -> (B, H, n_t * n_h * n_w * tile_t * tile_h * tile_w, D)
    x = x.view(B, H, S, D)
    return x


def untile(x: torch.Tensor, canvas_thw: Tuple[int, int, int],
           tile_thw: Tuple[int, int, int]) -> torch.Tensor:
    """Reverse the tiling operation to restore original token layout.

    Args:
        x: (B, H, S, D) in tile order
        canvas_thw: (T, H, W) spatiotemporal dimensions
        tile_thw: (tile_t, tile_h, tile_w) tile dimensions

    Returns:
        (B, H, S, D) in original (T, H, W) raster order
    """
    B, H, S, D = x.shape
    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw

    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w

    # (B, H, S, D) -> (B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D)
    x = x.view(B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D)
    # -> (B, H, n_t, tile_t, n_h, tile_h, n_w, tile_w, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).contiguous()
    # -> (B, H, S, D)
    x = x.view(B, H, S, D)
    return x


# ============================================================================
# STA (Sliding Tile Attention) mask
# ============================================================================

@lru_cache(maxsize=4096)
def _create_sta_3d_mask_cached(canvas_str: str, tile_str: str,
                               kernel_str: str) -> torch.Tensor:
    """Create STA mask using vectorized numpy operations (cached).

    Returns a boolean mask of shape (block_num, block_num) where
    mask[i, j] = True means query-tile i should attend to kv-tile j.
    """
    canvas_thw = tuple(map(int, canvas_str.split('_')))
    tile_thw = tuple(map(int, tile_str.split('_')))
    kernel_thw = tuple(map(int, kernel_str.split('_')))

    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    kernel_t, kernel_h, kernel_w = kernel_thw

    n_t = t // tile_t
    n_h = h // tile_h
    n_w = w // tile_w
    block_num = n_t * n_h * n_w

    i_indices = np.arange(block_num)
    j_indices = np.arange(block_num)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')

    # Decompose flat tile indices into (t, h, w) tile coordinates
    q_t = i_grid // (n_h * n_w)
    q_h = (i_grid % (n_h * n_w)) // n_w
    q_w = i_grid % n_w

    kv_t = j_grid // (n_h * n_w)
    kv_h = (j_grid % (n_h * n_w)) // n_w
    kv_w = j_grid % n_w

    # Clamp kernel center to valid range (handles boundary tiles)
    center_t = np.clip(q_t, kernel_t // 2, (n_t - 1) - kernel_t // 2)
    center_h = np.clip(q_h, kernel_h // 2, (n_h - 1) - kernel_h // 2)
    center_w = np.clip(q_w, kernel_w // 2, (n_w - 1) - kernel_w // 2)

    # Check if kv tile is within the kernel window
    time_mask = np.abs(center_t - kv_t) <= kernel_t // 2
    height_mask = np.abs(center_h - kv_h) <= kernel_h // 2
    width_mask = np.abs(center_w - kv_w) <= kernel_w // 2

    block_mask = time_mask & height_mask & width_mask
    return torch.tensor(block_mask, dtype=torch.bool)


def create_sta_3d_mask(canvas_thw: Tuple[int, int, int],
                       tile_thw: Tuple[int, int, int],
                       kernel_thw: Tuple[int, int, int],
                       text_block_num: int = 0) -> torch.Tensor:
    """Create STA (Sliding Tile Attention) 3D block mask.

    Args:
        canvas_thw: (T, H, W) spatiotemporal dimensions
        tile_thw: (tile_t, tile_h, tile_w) tile dimensions
        kernel_thw: (kernel_t, kernel_h, kernel_w) local window in tile units
        text_block_num: number of text blocks to append

    Returns:
        Boolean mask of shape (total_blocks, total_blocks)
    """
    canvas_str = '_'.join(str(x) for x in canvas_thw)
    tile_str = '_'.join(str(x) for x in tile_thw)
    kernel_str = '_'.join(str(x) for x in kernel_thw)

    block_mask = _create_sta_3d_mask_cached(canvas_str, tile_str, kernel_str)
    block_num = block_mask.shape[0]

    if text_block_num > 0:
        total = block_num + text_block_num
        sta_mask = torch.full((total, total), False, dtype=torch.bool)
        sta_mask[:block_num, :block_num] = block_mask
        # Text blocks attend to everything and are attended by everything
        sta_mask[:, -text_block_num:] = True
        sta_mask[-text_block_num:, :] = True
        return sta_mask
    return block_mask


# ============================================================================
# Importance sampling for selective attention
# ============================================================================

def importance_sampling(q: torch.Tensor, k: torch.Tensor,
                        topk: int, lambda_: float = 0.7
                        ) -> torch.Tensor:
    """Select top-k KV blocks based on importance scores.

    Importance = lambda * similarity(q, k) - (1 - lambda) * redundancy(k, k)

    Args:
        q: (B_or_1, H_or_1, num_q_blocks, D) block-averaged query features
        k: (B_or_1, H_or_1, num_kv_blocks, D) block-averaged key features
        topk: number of blocks to select per query block
        lambda_: weight balancing similarity vs redundancy

    Returns:
        top_block_indices: (B_or_1, H_or_1, num_q_blocks, topk)
    """
    # Normalize
    q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    # Similarity: (B, H, Q_blocks, K_blocks)
    similarity = torch.einsum("bhsd,bhkd->bhsk", q_norm, k_norm)
    similarity = (similarity + 1.0) / 2.0  # map from [-1,1] to [0,1]

    # Redundancy among keys: (B, H, K_blocks, K_blocks)
    key_sim = torch.einsum("bhsd,bhkd->bhsk", k_norm, k_norm)
    key_sim = (key_sim + 1.0) / 2.0

    # Mask diagonal (self-similarity)
    K_num = k_norm.shape[2]
    diag = torch.arange(K_num, device=k.device)
    key_sim[:, :, diag, diag] = float('nan')

    # Mean redundancy per key block: (B, H, 1, K_blocks)
    mean_redundancy = torch.nanmean(key_sim, dim=-2, keepdim=True)

    # Importance scores: (B, H, Q_blocks, K_blocks)
    scores = lambda_ * similarity - (1 - lambda_) * mean_redundancy

    topk = min(topk, scores.shape[-1])
    _, indices = scores.topk(k=topk, dim=-1, sorted=False)
    return indices


def similarity_sampling(q: torch.Tensor, k: torch.Tensor,
                        topk: int) -> torch.Tensor:
    """Select top-k KV blocks based on simple similarity scores.

    Args:
        q: (B_or_1, H_or_1, num_q_blocks, D) block-averaged query features
        k: (B_or_1, H_or_1, num_kv_blocks, D) block-averaged key features
        topk: number of blocks to select per query block

    Returns:
        top_block_indices: (B_or_1, H_or_1, num_q_blocks, topk)
    """
    gate = torch.einsum("bhsd,bhkd->bhsk", q, k)
    topk = min(topk, gate.shape[-1])
    _, indices = gate.topk(k=topk, dim=-1, sorted=False)
    return indices


# ============================================================================
# SSTA mask creation (STA + selective)
# ============================================================================

def create_moba_3d_mask(q: torch.Tensor, k: torch.Tensor,
                        canvas_thw: Tuple[int, int, int],
                        topk: int,
                        tile_thw: Tuple[int, int, int],
                        text_block_num: int = 0,
                        add_text_mask: bool = False,
                        lambda_: float = 0.7,
                        mask_share_within_head: bool = True,
                        sampling_type: str = 'importance'
                        ) -> torch.Tensor:
    """Create selective (MOBA) block attention mask via importance/similarity sampling.

    Args:
        q: (B, H, S_image, D) image query tokens in tile order
        k: (B, H, S_image, D) image key tokens in tile order
        canvas_thw: (T, H, W) spatiotemporal dimensions
        topk: number of non-local blocks to select
        tile_thw: tile dimensions
        text_block_num: number of text blocks
        add_text_mask: whether to add text-to-all attention
        lambda_: importance sampling weight
        mask_share_within_head: if True, average across heads before sampling
        sampling_type: 'importance' or 'similarity'

    Returns:
        Boolean mask of shape (num_heads_or_1, total_blocks, total_blocks)
    """
    t, h, w = canvas_thw
    tile_t, tile_h, tile_w = tile_thw
    block_size = tile_t * tile_h * tile_w
    n_t, n_h, n_w = t // tile_t, h // tile_h, w // tile_w
    block_num = n_t * n_h * n_w

    B, H, S, D = k.shape

    # Compute block-averaged features
    k_blocks = k.view(B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D)
    k_avg = k_blocks.mean(dim=(-2, -3, -4)).view(B, H, -1, D)

    q_blocks = q.view(B, H, n_t, n_h, n_w, tile_t, tile_h, tile_w, D)
    q_avg = q_blocks.mean(dim=(-2, -3, -4)).view(B, H, -1, D)

    # Cast to float32 for better scoring precision
    q_avg = q_avg.float()
    k_avg = k_avg.float()

    if mask_share_within_head:
        q_avg = q_avg.mean(dim=1, keepdim=True)
        k_avg = k_avg.mean(dim=1, keepdim=True)

    # Select top-k blocks
    if sampling_type == 'importance':
        top_indices = importance_sampling(q_avg, k_avg, topk, lambda_)
    elif sampling_type == 'similarity':
        top_indices = similarity_sampling(q_avg, k_avg, topk)
    else:
        raise ValueError(f"Unsupported sampling_type: {sampling_type}")

    # Build mask from indices
    assert top_indices.shape[0] == 1, "batch size must be 1 for mask creation"
    top_indices = top_indices.squeeze(0)  # (H_or_1, block_num, topk)

    num_heads_dim = top_indices.shape[0]
    gate_mask = torch.zeros(num_heads_dim, block_num, block_num,
                            dtype=torch.bool, device=q.device)

    for h_idx in range(num_heads_dim):
        gate_mask[h_idx].scatter_(dim=-1, index=top_indices[h_idx], value=True)

    # Pad with text blocks
    if text_block_num > 0:
        total = block_num + text_block_num
        full_mask = torch.full((num_heads_dim, total, total), False,
                               dtype=torch.bool, device=q.device)
        full_mask[:, :block_num, :block_num] = gate_mask
        if add_text_mask:
            full_mask[:, :, -text_block_num:] = True
            full_mask[:, -text_block_num:, :] = True
        return full_mask

    return gate_mask


@torch.no_grad()
def create_ssta_3d_mask(q: torch.Tensor, k: torch.Tensor,
                        canvas_thw: Tuple[int, int, int],
                        topk: int,
                        tile_thw: Tuple[int, int, int],
                        kernel_thw: Tuple[int, int, int],
                        text_block_num: int = 0,
                        lambda_: float = 0.7,
                        text_mask: Optional[torch.Tensor] = None,
                        mask_share_within_head: bool = True,
                        sampling_type: str = 'importance'
                        ) -> torch.Tensor:
    """Create combined SSTA mask = STA (local) OR MOBA (selective).

    Args:
        q, k: image tokens in tile order (1, H, S_image, D)
        canvas_thw: (T, H, W)
        topk: number of non-local blocks to select
        tile_thw: tile dimensions
        kernel_thw: STA kernel dimensions
        text_block_num: number of text blocks
        lambda_: importance sampling weight
        text_mask: optional per-token text mask
        mask_share_within_head: share mask across heads
        sampling_type: 'importance' or 'similarity'

    Returns:
        Boolean mask of shape (num_heads_or_1, total_blocks, total_blocks)
    """
    # Fixed local window mask
    sta_mask = create_sta_3d_mask(canvas_thw, tile_thw, kernel_thw,
                                 text_block_num).to(q.device)

    # Selective non-local mask
    moba_mask = create_moba_3d_mask(
        q, k, canvas_thw, topk, tile_thw,
        text_block_num=text_block_num,
        add_text_mask=True,
        lambda_=lambda_,
        mask_share_within_head=mask_share_within_head,
        sampling_type=sampling_type,
    )

    # Combine: attend if either local OR selected
    ssta_mask = torch.logical_or(sta_mask.unsqueeze(0), moba_mask)

    # Apply text padding mask
    if text_mask is not None:
        block_size = int(np.prod(tile_thw))
        seq_len = q.shape[2]
        block_num = seq_len // block_size
        text_mask_index = int(math.ceil(text_mask.sum().item() / block_size))
        text_mask_index = max(text_mask_index, 1)

        pad_start = block_num + text_mask_index
        total = ssta_mask.shape[-1]
        if pad_start < total:
            ssta_mask[:, pad_start:, :] = False
            ssta_mask[:, :, pad_start:] = False
            # Self-attention for padded text blocks
            pad_size = total - pad_start
            eye = torch.eye(pad_size, dtype=torch.bool,
                            device=ssta_mask.device).unsqueeze(0)
            ssta_mask[:, pad_start:, pad_start:] = (
                ssta_mask[:, pad_start:, pad_start:] | eye
            )

    return ssta_mask


# ============================================================================
# Block mask -> token-level mask expansion
# ============================================================================

def expand_block_mask(block_mask: torch.Tensor,
                      block_size: int) -> torch.Tensor:
    """Expand a block-level boolean mask to a token-level mask.

    Args:
        block_mask: (num_heads_or_1, num_blocks, num_blocks) boolean
        block_size: number of tokens per block

    Returns:
        Token-level mask of shape (num_heads_or_1, S, S) where
        S = num_blocks * block_size. True = attend, False = mask out.
    """
    # Use Kronecker product: each True block becomes a block_size x block_size
    # all-True region
    H, NB, NB2 = block_mask.shape
    assert NB == NB2
    # (H, NB, NB) -> (H, NB, 1, NB, 1) -> (H, NB, BS, NB, BS) -> (H, S, S)
    token_mask = block_mask[:, :, None, :, None].expand(
        H, NB, block_size, NB, block_size
    ).reshape(H, NB * block_size, NB * block_size)
    return token_mask


# ============================================================================
# Main SSTA 3D Attention operator
# ============================================================================

class Model(nn.Module):
    """
    Selective and Sliding Tile Attention (SSTA) for 3D video tokens.

    Combines a fixed local 3D tile window (STA) with data-dependent
    non-local tile selection (importance/similarity sampling).

    This is a pure PyTorch reference implementation. The Tencent production
    code uses flex_block_attn_func for the block-sparse attention kernel.
    The intent is for users to replace the attention computation with an
    optimized kernel while keeping the mask generation logic.

    Args:
        tile_size: (tile_t, tile_h, tile_w) tile dimensions
        kernel_size: (kernel_t, kernel_h, kernel_w) STA window in tile units
        topk: number of non-local tiles to select per query tile
        lambda_: weight for importance sampling (similarity vs redundancy)
        sampling_type: 'importance' or 'similarity'
        threshold: threshold for dynamic topk (0.0 = fixed topk)
        pad_type: 'zero' or 'repeat' for padding to tile boundaries
    """

    def __init__(self,
                 tile_size: Tuple[int, int, int] = (6, 8, 8),
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 topk: int = 64,
                 lambda_: float = 0.7,
                 sampling_type: str = 'importance',
                 threshold: float = 0.0,
                 pad_type: str = 'zero'):
        super().__init__()
        self.tile_size = tile_size
        self.kernel_size = kernel_size
        self.topk = topk
        self.lambda_ = lambda_
        self.sampling_type = sampling_type
        self.threshold = threshold
        self.pad_type = pad_type

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                canvas_thw: Tuple[int, int, int],
                text_len: int = 0,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SSTA 3D attention.

        Args:
            q: (B, H, S, D) query tensor where S = T*H*W + text_len
            k: (B, H, S, D) key tensor
            v: (B, H, S, D) value tensor
            canvas_thw: (T, H, W) spatiotemporal grid dimensions of video tokens
            text_len: number of text tokens at the end of the sequence
            text_mask: optional (B, text_len) boolean mask for text tokens

        Returns:
            (B, H, S, D) attention output
        """
        B, H, S, D = q.shape
        t, h, w = canvas_thw
        tile_t, tile_h, tile_w = self.tile_size
        block_size = tile_t * tile_h * tile_w

        # ---- Separate image and text tokens ----
        if text_len > 0:
            image_q = q[:, :, :-text_len, :]
            image_k = k[:, :, :-text_len, :]
            image_v = v[:, :, :-text_len, :]
            text_q = q[:, :, -text_len:, :]
            text_k = k[:, :, -text_len:, :]
            text_v = v[:, :, -text_len:, :]
        else:
            image_q, image_k, image_v = q, k, v

        # ---- Pad image tokens to tile boundaries ----
        need_pad = False
        pad_t = (tile_t - t % tile_t) % tile_t
        pad_h = (tile_h - h % tile_h) % tile_h
        pad_w = (tile_w - w % tile_w) % tile_w

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            need_pad = True
            image_q = image_q.view(B, H, t, h, w, D)
            image_k = image_k.view(B, H, t, h, w, D)
            image_v = image_v.view(B, H, t, h, w, D)

            if pad_t > 0:
                pad_val = torch.zeros_like if self.pad_type == 'zero' else lambda x: x
                pq = torch.zeros(B, H, pad_t, h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_q[:, :, -1:].expand(-1, -1, pad_t, -1, -1, -1)
                pk = torch.zeros(B, H, pad_t, h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_k[:, :, -1:].expand(-1, -1, pad_t, -1, -1, -1)
                pv = torch.zeros(B, H, pad_t, h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_v[:, :, -1:].expand(-1, -1, pad_t, -1, -1, -1)
                image_q = torch.cat([image_q, pq], dim=2)
                image_k = torch.cat([image_k, pk], dim=2)
                image_v = torch.cat([image_v, pv], dim=2)
                t = t + pad_t

            if pad_h > 0:
                pq = torch.zeros(B, H, t, pad_h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_q[:, :, :, -1:].expand(-1, -1, -1, pad_h, -1, -1)
                pk = torch.zeros(B, H, t, pad_h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_k[:, :, :, -1:].expand(-1, -1, -1, pad_h, -1, -1)
                pv = torch.zeros(B, H, t, pad_h, w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_v[:, :, :, -1:].expand(-1, -1, -1, pad_h, -1, -1)
                image_q = torch.cat([image_q, pq], dim=3)
                image_k = torch.cat([image_k, pk], dim=3)
                image_v = torch.cat([image_v, pv], dim=3)
                h = h + pad_h

            if pad_w > 0:
                pq = torch.zeros(B, H, t, h, pad_w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_q[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_w, -1)
                pk = torch.zeros(B, H, t, h, pad_w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_k[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_w, -1)
                pv = torch.zeros(B, H, t, h, pad_w, D, device=q.device, dtype=q.dtype) if self.pad_type == 'zero' else image_v[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_w, -1)
                image_q = torch.cat([image_q, pq], dim=4)
                image_k = torch.cat([image_k, pk], dim=4)
                image_v = torch.cat([image_v, pv], dim=4)
                w = w + pad_w

            image_q = image_q.reshape(B, H, -1, D)
            image_k = image_k.reshape(B, H, -1, D)
            image_v = image_v.reshape(B, H, -1, D)

        padded_canvas = (t, h, w)

        # ---- Pad text tokens to block boundary ----
        need_pad_text = False
        text_block_num = math.ceil(text_len / block_size) if text_len > 0 else 0
        text_target_size = text_block_num * block_size
        text_pad_size = 0

        if text_len > 0 and text_len % block_size > 0:
            need_pad_text = True
            text_pad_size = text_target_size - text_len
            pad_tq = text_q[:, :, -1:].expand(-1, -1, text_pad_size, -1)
            pad_tk = text_k[:, :, -1:].expand(-1, -1, text_pad_size, -1)
            pad_tv = text_v[:, :, -1:].expand(-1, -1, text_pad_size, -1)
            text_q = torch.cat([text_q, pad_tq], dim=2)
            text_k = torch.cat([text_k, pad_tk], dim=2)
            text_v = torch.cat([text_v, pad_tv], dim=2)

        # ---- Tile image tokens ----
        image_q = tile(image_q, padded_canvas, self.tile_size)
        image_k = tile(image_k, padded_canvas, self.tile_size)
        image_v = tile(image_v, padded_canvas, self.tile_size)

        # ---- Concatenate image + text ----
        if text_len > 0:
            tiled_q = torch.cat([image_q, text_q], dim=2)
            tiled_k = torch.cat([image_k, text_k], dim=2)
            tiled_v = torch.cat([image_v, text_v], dim=2)
        else:
            tiled_q, tiled_k, tiled_v = image_q, image_k, image_v

        # ---- Build SSTA block mask ----
        # Process each batch element separately (mask depends on data)
        mask_list = []
        for i in range(B):
            block_mask = create_ssta_3d_mask(
                image_q[i:i+1], image_k[i:i+1],
                canvas_thw=padded_canvas,
                topk=self.topk,
                tile_thw=self.tile_size,
                kernel_thw=self.kernel_size,
                text_block_num=text_block_num,
                lambda_=self.lambda_,
                text_mask=text_mask[i] if text_mask is not None else None,
                mask_share_within_head=True,
                sampling_type=self.sampling_type,
            )
            mask_list.append(block_mask)

        # (B, 1, total_blocks, total_blocks)
        block_mask = torch.stack(mask_list, dim=0)

        # ---- Expand block mask to token-level mask ----
        # block_mask: (B, 1, NB, NB) -> token mask: (B, 1, S_tiled, S_tiled)
        token_mask = expand_block_mask(
            block_mask.view(-1, block_mask.shape[-2], block_mask.shape[-1]),
            block_size
        ).view(B, 1, -1, tiled_q.shape[2])

        # Convert bool mask to float mask for SDPA: True -> 0.0, False -> -inf
        float_mask = torch.where(
            token_mask,
            torch.tensor(0.0, device=q.device, dtype=q.dtype),
            torch.tensor(float('-inf'), device=q.device, dtype=q.dtype),
        )

        # ---- Compute attention ----
        attn_output = F.scaled_dot_product_attention(
            tiled_q, tiled_k, tiled_v,
            attn_mask=float_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # ---- Separate and untile ----
        if text_len > 0:
            image_o = attn_output[:, :, :-text_target_size, :]
            if need_pad_text:
                text_o = attn_output[:, :, -text_target_size:-text_pad_size, :]
            else:
                text_o = attn_output[:, :, -text_target_size:, :]
        else:
            image_o = attn_output

        # Untile image tokens
        image_o = untile(image_o, padded_canvas, self.tile_size)

        # Remove spatial padding
        if need_pad:
            image_o = image_o.view(B, H, t, h, w, D)
            orig_t = t - pad_t if pad_t > 0 else t
            orig_h = h - pad_h if pad_h > 0 else h
            orig_w = w - pad_w if pad_w > 0 else w
            image_o = image_o[:, :, :orig_t, :orig_h, :orig_w, :]
            image_o = image_o.reshape(B, H, -1, D)

        # Recombine image + text
        if text_len > 0:
            output = torch.cat([image_o, text_o], dim=2)
        else:
            output = image_o

        return output
