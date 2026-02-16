"""
Fast-dLLM (Fast Diffusion Large Language Model)

Implements the LLaDA masked diffusion language model backbone and the
Fast-dLLM inference acceleration framework:
- LLaDA backbone: Qwen2.5-based transformer with bidirectional (non-causal) attention
- Block-wise masked diffusion decoding (vanilla, prefix-cache, dual-cache)
- Confidence-aware parallel decoding with threshold-based token unmasking
- Gumbel-max sampling for categorical distributions

Reference: https://github.com/NVlabs/Fast-dLLM
Paper: https://arxiv.org/abs/2505.22618 (v1), https://arxiv.org/abs/2509.26328 (v2)
Backbone: GSAI-ML/LLaDA-8B-Instruct (https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)

Architecture (LLaDA backbone):
    input_ids -> Embedding -> N x [RMSNorm -> BidirectionalAttention -> RMSNorm -> SwiGLU MLP]
    -> RMSNorm -> lm_head -> logits

Key difference from autoregressive models: attention is BIDIRECTIONAL (no causal mask).
The model predicts masked tokens ([MASK] id=126336) iteratively via diffusion.

Generation (Fast-dLLM):
    1. generate(): vanilla block-wise MDM decoding
    2. generate_with_prefix_cache(): + KV cache for prefix tokens
    3. generate_with_dual_cache(): + cache for masked suffix positions

This model uses level1 operators from KernelBench.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNormL1
from ..level1.activations._7_Swish import Model as Swish


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "LLaDA-8B-Instruct": {
        "model_name": "GSAI-ML/LLaDA-8B-Instruct",
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 32,
        "n_layers": 32,
        "mlp_hidden_size": 12288,
        "vocab_size": 126464,
        "embedding_size": 126464,
        "mask_token_id": 126336,
        "max_sequence_length": 4096,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "weight_tying": False,
    },
}


# ============================================================================
# Wrapper classes for level1 operators
# ============================================================================

class RMSNorm(nn.Module):
    """RMS normalization with learnable weight, using level1 operator."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self._rms_norm = RMSNormL1(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._rms_norm(x.transpose(1, -1)).transpose(1, -1)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding matching the reference LLaDA implementation.

    Keeps cos/sin cache in float32 at all times (immune to model.to(bfloat16)),
    applies RoPE at float32 precision, then casts back. This matches the
    reference's rope_full_precision=True behavior exactly.

    Handles the case where q and k have different sequence lengths (KV cache)
    and dual-cache block_end_index for position computation.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._max_cached = 0
        # A dummy buffer for device tracking (survives .to() calls)
        self.register_buffer("_device_tracker", torch.zeros(1), persistent=False)
        # Pre-build cache on CPU (matching reference behavior)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build float32 cos/sin cache on CPU.

        The reference computes inv_freq and cos/sin on CPU during model
        construction, then moves them to GPU via .to(device). We match this
        exactly by always computing on CPU and moving the result.
        """
        from torch import einsum as _einsum
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = _einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self._pos_cos = emb.cos()  # (seq_len, head_dim), float32, CPU
        self._pos_sin = emb.sin()  # (seq_len, head_dim), float32, CPU
        self._max_cached = seq_len

    def _ensure_cache(self, seq_len: int, device: torch.device = None):
        """Extend the float32 cos/sin cache to cover at least *seq_len* positions."""
        if seq_len <= self._max_cached:
            # Move to device if needed
            if device is not None and self._pos_cos.device != device:
                self._pos_cos = self._pos_cos.to(device)
                self._pos_sin = self._pos_sin.to(device)
            return
        self._build_cache(seq_len)
        # Move to target device
        if device is not None:
            self._pos_cos = self._pos_cos.to(device)
            self._pos_sin = self._pos_sin.to(device)

    def _apply(self, fn):
        """Override to keep cos/sin cache in float32 when model dtype changes.

        The cache is always computed on CPU (matching the reference), then
        moved to the target device. We don't recompute here since _ensure_cache
        handles device movement lazily.
        """
        super()._apply(fn)
        # Move existing cache to the new device (keeping float32)
        device = self._device_tracker.device
        if self._max_cached > 0:
            self._pos_cos = self._pos_cos.to(device=device, dtype=torch.float32)
            self._pos_sin = self._pos_sin.to(device=device, dtype=torch.float32)
        return self

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                block_end_index: Optional[int] = None) -> tuple:
        """
        Apply RoPE to q and k tensors.

        Args:
            q: (B, n_heads, q_len, head_dim)
            k: (B, n_kv_heads, k_len, head_dim)
            block_end_index: When set (dual-cache mode), queries get positions
                [block_end_index - q_len, block_end_index) instead of the
                default [k_len - q_len, k_len).
        """
        orig_dtype = q.dtype
        q_len = q.shape[2]
        k_len = k.shape[2]

        # Determine query position range
        if block_end_index is not None:
            q_start = block_end_index - q_len
        else:
            q_start = k_len - q_len
        q_end = q_start + q_len

        # Ensure cache is large enough and on the right device
        max_pos = max(q_end, k_len)
        self._ensure_cache(max_pos, q.device)

        # Slice cos/sin for q and k positions (always float32)
        q_cos = self._pos_cos[q_start:q_end].unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, head_dim)
        q_sin = self._pos_sin[q_start:q_end].unsqueeze(0).unsqueeze(0)
        k_cos = self._pos_cos[:k_len].unsqueeze(0).unsqueeze(0)  # (1, 1, k_len, head_dim)
        k_sin = self._pos_sin[:k_len].unsqueeze(0).unsqueeze(0)

        # Cast q, k to float32 for full-precision RoPE
        q_fp32 = q.float()
        k_fp32 = k.float()

        # Apply rotary embedding
        q_rotated = (q_fp32 * q_cos) + (self._rotate_half(q_fp32) * q_sin)
        k_rotated = (k_fp32 * k_cos) + (self._rotate_half(k_fp32) * k_sin)

        return q_rotated.to(orig_dtype), k_rotated.to(orig_dtype)


# ============================================================================
# Compiled SDPA (matching reference's @torch.compile wrapper)
# ============================================================================

@torch.compile()
def _compiled_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
    )


# ============================================================================
# Component Modules
# ============================================================================

class LLaDAAttention(nn.Module):
    """
    Bidirectional self-attention with optional GQA and RoPE.

    Critical difference from autoregressive models: NO causal mask.
    All tokens attend to all other tokens (bidirectional).
    Supports KV cache for Fast-dLLM prefix/dual cache generation.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float = 500000.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_kv_groups = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache BEFORE RoPE (matching reference: RoPE is re-applied
        # to ALL keys including cached ones for bidirectional attention)
        rope_block_end = None
        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                for b in range(batch_size):
                    idx = replace_position[b].nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        past_key[b, :, idx] = k[b, :, :len(idx)]
                        past_value[b, :, idx] = v[b, :, :len(idx)]
                k = past_key
                v = past_value
                if replace_position.any():
                    rope_block_end = int(replace_position.nonzero(as_tuple=True)[1].max().item()) + 1

        present = (k, v) if use_cache else None

        # Apply RoPE after KV cache concatenation. For bidirectional attention,
        # all keys (including cached) get fresh positional encoding every time.
        q, k = self.rotary_emb(q, k, block_end_index=rope_block_end)

        # GQA/MHA: expand KV heads if needed (SDPA doesn't natively support GQA)
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Scaled dot-product attention -- BIDIRECTIONAL (no causal mask)
        # Uses @torch.compile wrapper matching the reference implementation
        attn_output = _compiled_sdpa(q, k, v, dropout_p=0.0, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), present


class LLaDAMLP(nn.Module):
    """
    SiLU-gated MLP (gate_proj + up_proj + down_proj).

    Matches the reference LLaDALlamaBlock MLP:
      silu(gate_proj(x)) * up_proj(x) -> down_proj
    """
    def __init__(self, d_model: int, mlp_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(d_model, mlp_hidden_size, bias=False)
        self.down_proj = nn.Linear(mlp_hidden_size, d_model, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class LLaDABlock(nn.Module):
    """LLaDA transformer block (pre-norm, LLaMA-style)."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_hidden_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, rms_norm_eps)
        self.self_attn = LLaDAAttention(d_model, n_heads, n_kv_heads, rope_theta, max_seq_len)
        self.post_attention_layernorm = RMSNorm(d_model, rms_norm_eps)
        self.mlp = LLaDAMLP(d_model, mlp_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.input_layernorm(x)
        x, present = self.self_attn(x, layer_past, use_cache, replace_position)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Fast-dLLM: LLaDA masked diffusion LM backbone + Fast-dLLM generation.

    The backbone is a Qwen2.5-based transformer with BIDIRECTIONAL attention
    (no causal mask). The model predicts masked tokens iteratively.

    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - Swish/SiLU from level1/activations/7_Swish

    Supports variants: LLaDA-8B-Instruct
    """

    VARIANTS = VARIANTS

    def __init__(self, **kwargs):
        super().__init__()

        # Match reference SDPA backend selection
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.d_model = kwargs.get("d_model", 4096)
        self.n_heads = kwargs.get("n_heads", 32)
        self.n_kv_heads = kwargs.get("n_kv_heads", 32)
        self.n_layers = kwargs.get("n_layers", 32)
        self.mlp_hidden_size = kwargs.get("mlp_hidden_size", 12288)
        self.vocab_size = kwargs.get("vocab_size", 126464)
        self.embedding_size = kwargs.get("embedding_size", self.vocab_size)
        self.mask_token_id = kwargs.get("mask_token_id", 126336)
        self.max_sequence_length = kwargs.get("max_sequence_length", 4096)
        self.rope_theta = kwargs.get("rope_theta", 500000.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-5)
        self.weight_tying = kwargs.get("weight_tying", False)

        # Embedding (embedding_size may differ from vocab_size)
        self.embed_tokens = nn.Embedding(self.embedding_size, self.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            LLaDABlock(
                self.d_model, self.n_heads, self.n_kv_heads,
                self.mlp_hidden_size, self.rms_norm_eps,
                self.rope_theta, self.max_sequence_length,
            )
            for _ in range(self.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(self.d_model, self.rms_norm_eps)

        # LM head (may be tied to embedding weights)
        if self.weight_tying:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(self.d_model, self.embedding_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token ids
            past_key_values: optional list of (key, value) tuples per layer
            use_cache: whether to return updated KV cache
            replace_position: (batch, full_seq_len) bool mask for dual-cache position replacement

        Returns:
            An object with .logits and optionally .past_key_values
        """
        x = self.embed_tokens(input_ids)

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, layer_past, use_cache, replace_position)
            if use_cache:
                new_past_key_values.append(present)

        x = self.norm(x)

        if self.weight_tying:
            logits = F.linear(x, self.embed_tokens.weight, None)
        else:
            logits = self.lm_head(x)

        return _ModelOutput(logits=logits, past_key_values=new_past_key_values)


class _ModelOutput:
    """Simple container matching transformers CausalLMOutputWithPast interface."""
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


# ============================================================================
# Fast-dLLM Generation Utilities
# ============================================================================

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for sampling categorical distributions.

    When temperature=0, returns logits unchanged (greedy).
    Uses float64 for precision as recommended by arXiv:2409.02908.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    block_mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    Precompute how many tokens to unmask at each denoising step.

    Distributes masked tokens evenly across steps, with remainder
    going to the first few steps.

    Args:
        block_mask_index: (B, block_len) bool mask of which positions are masked
        steps: number of denoising steps

    Returns:
        (B, steps) int tensor of tokens to transfer per step
    """
    device = block_mask_index.device

    total = block_mask_index.sum(dim=1)  # (B,)
    base = torch.div(total, steps, rounding_mode="floor")  # (B,)
    rem = total - base * steps  # (B,)

    num_transfer = base.unsqueeze(1).expand(-1, steps).to(torch.long)

    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    num_transfer = num_transfer + add_mask.to(torch.long)

    return num_transfer


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select which masked positions to unmask this step.

    Args:
        logits: (B, L, V) model output logits
        temperature: sampling temperature (0 = greedy)
        remasking: 'low_confidence' or 'random'
        mask_index: (B, L) bool mask of currently masked positions
        x: (B, L) current token ids
        num_transfer_tokens: (B,) how many tokens to unmask (None if using threshold)
        threshold: confidence threshold for parallel decoding (None if using fixed count)

    Returns:
        x0: (B, L) proposed token ids
        transfer_index: (B, L) bool mask of positions to unmask
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


# ============================================================================
# Fast-dLLM Generation Functions
# ============================================================================

@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Vanilla block-wise masked diffusion decoding (no cache).

    Args:
        model: LLaDA model (forward returns .logits)
        prompt: (1, L) prompt token ids
        steps: total denoising steps
        gen_length: number of tokens to generate
        block_length: block size for semi-autoregressive decoding
        temperature: sampling temperature (0 = greedy)
        remasking: 'low_confidence' or 'random'
        mask_id: mask token id (126336 for LLaDA)
        threshold: confidence threshold for parallel decoding

    Returns:
        x: (1, L + gen_length) full sequence with generated tokens
        nfe: number of forward evaluations
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long, device=prompt.device,
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, block_end:] = 0

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index, x,
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
            i += 1

            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Block-wise decoding with KV cache for prefix tokens.

    After the first forward pass on the full sequence, subsequent passes
    only process the current block, reusing cached KVs for the prefix.

    Returns:
        x: (1, L + gen_length) full sequence with generated tokens
        nfe: number of forward evaluations
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long, device=prompt.device,
    )
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # First forward: full sequence, build KV cache
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0

        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None,
            threshold,
        )
        x[transfer_index] = x0[transfer_index]

        # Trim KV cache to prefix only
        new_past = []
        for layer_kv in past_key_values:
            new_past.append(
                tuple(kv[:, :, :block_start] for kv in layer_kv)
            )
        past_key_values = new_past
        nfe += 1

        i = 1
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            nfe += 1

            mask_index = (x[:, block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(
                x[:, block_start:],
                past_key_values=past_key_values,
                use_cache=True,
            ).logits

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index,
                x[:, block_start:],
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
            )
            x[:, block_start:][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Block-wise decoding with dual KV cache (prefix + suffix).

    In addition to caching prefix KVs, also caches and updates KVs
    for masked suffix positions using replace_position.

    Returns:
        x: (1, L + gen_length) full sequence with generated tokens
        nfe: number of forward evaluations
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full(
        (B, Lp + gen_length), mask_id, dtype=torch.long, device=prompt.device,
    )
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Full forward pass with cache
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # Step 0: initial transfer
        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        x0, transfer_index = get_transfer_index(
            out_full.logits, temperature, remasking, global_mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None,
            threshold,
        )
        x = torch.where(transfer_index, x0, x)

        # Refinement steps
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            logits_blk = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits

            mask_blk = (x[:, s:e] == mask_id)

            x0_blk, transfer_idx_blk = get_transfer_index(
                logits_blk, temperature, remasking, mask_blk, x[:, s:e],
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
            )

            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
            nfe += 1

    return x, nfe


    return x, nfe
