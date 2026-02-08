"""
RetNet (Retentive Network) Model

Implements RetNet architecture following the flash-linear-attention (fla) library:
- Multi-scale retention with rotary position embeddings
- Fixed per-head exponential decay (gamma)
- SwiGLU feed-forward network
- Compatible with fla-hub/retnet-* HuggingFace checkpoints

Variants:
- RetNet-1.3B: 24 layers, hidden_size=2048, num_heads=8
- RetNet-2.7B: 32 layers, hidden_size=2560, num_heads=10

HF weight layout (prefix: model.):
  model.embeddings.weight
  model.layers[i].attn_norm.weight
  model.layers[i].attn.q_proj.weight
  model.layers[i].attn.k_proj.weight
  model.layers[i].attn.v_proj.weight
  model.layers[i].attn.g_proj.weight
  model.layers[i].attn.o_proj.weight
  model.layers[i].attn.g_norm.weight
  model.layers[i].mlp_norm.weight
  model.layers[i].mlp.gate_proj.weight
  model.layers[i].mlp.up_proj.weight
  model.layers[i].mlp.down_proj.weight
  model.norm.weight
  lm_head.weight

This model uses level1 operators from KernelBench.
It does NOT depend on flash-linear-attention at runtime.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.activations._7_Swish import Model as Swish
from ..level1.embeddings._2_Embedding import Model as Embedding

# Import level1 SSM operators
from ..level1.ssm._11_Retention import Model as RetentionRecurrence


# ============================================================================
# Rotary Embedding (matching fla's RotaryEmbedding exactly)
# ============================================================================

def _rotate_half(x):
    """Rotate the last dimension: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class SimpleRotaryEmbedding(nn.Module):
    """
    Rotary position embedding matching fla's RotaryEmbedding semantics.

    Uses the same inv_freq computation and cos/sin cache as fla.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        # Cached as full-dimension (T, D) tensors with broadcast shape (1, T, 1, D)
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seqlen: int, device, dtype):
        if seqlen <= self._seq_len_cached and self._cos_cached is not None:
            return
        self._seq_len_cached = seqlen
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        # Pre-compute full-dimension cos/sin with broadcast shape (1, T, 1, D)
        # so we avoid repeated cat+unsqueeze on every forward call.
        cos_half = freqs.cos().to(dtype)  # (T, D//2)
        sin_half = freqs.sin().to(dtype)  # (T, D//2)
        self._cos_cached = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        self._sin_cached = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)

    def forward(self, q, k, seqlen_offset=0, max_seqlen=None):
        """
        Apply rotary embeddings to q and k.

        Args:
            q: (B, T, H, D)
            k: (B, T, H, D)
            seqlen_offset: int offset for position indices
            max_seqlen: max sequence length for cache

        Returns:
            (q_rotated, k_rotated) with same shapes
        """
        T = q.shape[1]
        if max_seqlen is None:
            max_seqlen = T + seqlen_offset
        self._update_cache(max_seqlen, q.device, q.dtype)

        # Slice pre-computed full-dimension tensors: (1, T, 1, D)
        cos = self._cos_cached[:, seqlen_offset:seqlen_offset + T]
        sin = self._sin_cached[:, seqlen_offset:seqlen_offset + T]

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1.3B": "fla-hub/retnet-1.3B-100B",
    "2.7B": "fla-hub/retnet-2.7B-100B",
}


# ============================================================================
# RetNet Multi-Scale Retention
# ============================================================================

class MultiScaleRetention(nn.Module):
    """
    Multi-scale retention mechanism.

    Matches fla.layers.multiscale_retention.MultiScaleRetention architecture exactly.
    Uses rotary position embeddings and fixed per-head exponential decay.

    Uses level1 operators:
    - RetentionRecurrence for the retention recurrence
    - RotaryEmbedding for position encoding
    - RMSNorm for per-head output normalisation
    - Linear for projections
    - Swish for output gate activation
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        num_heads: int = 10,
        num_kv_heads: int = None,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Projections
        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        # Per-head group norm (RMSNorm with elementwise_affine per head_v_dim)
        self.g_norm = RMSNorm(
            self.head_v_dim,
            eps=norm_eps,
            learnable_weight=elementwise_affine,
            dim=-1,
        )

        # Rotary position embeddings
        self.rotary = SimpleRotaryEmbedding(dim=self.head_k_dim)

        # Level1 operators
        self.swish = Swish()

        # Level1 retention recurrent operator
        self.retention_recurrent = RetentionRecurrence(
            num_heads=num_heads,
            scale=None,
        )

        # Track sequence offset for autoregressive generation
        self._seqlen_offset = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head: (B, T, H, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_k_dim)

        # Compute position offset for autoregressive generation
        seqlen_offset = 0
        if cache_params is not None and cache_params.seqlen_offsets[self.layer_idx] is not None:
            seqlen_offset = cache_params.seqlen_offsets[self.layer_idx]

        # Apply rotary position embeddings
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset,
                           max_seqlen=seq_len + seqlen_offset)

        # Handle MQA/GQA: repeat kv heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_k_dim)

            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_v_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_v_dim)
        else:
            v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # Get initial recurrent state from cache (if available)
        initial_state = None
        if cache_params is not None:
            initial_state = cache_params.recurrent_states[self.layer_idx]

        # Run retention recurrence (level1 operator)
        o, final_state = self.retention_recurrent(
            q=q, k=k, v=v,
            initial_state=initial_state,
            output_final_state=use_cache,
        )

        # Save recurrent state and update offset
        if cache_params is not None:
            if final_state is not None:
                cache_params.update_recurrent_state(self.layer_idx, final_state)
            cache_params.update_seqlen_offset(self.layer_idx, seqlen_offset + seq_len)

        # Apply per-head RMSNorm: o is (B, T, H, V)
        o = self.g_norm(o)

        # Reshape to (B, T, H*V)
        o = o.reshape(batch_size, seq_len, -1)

        # Apply swish output gate
        g = self.g_proj(hidden_states)
        o = o * self.swish(g)

        # Output projection
        o = self.o_proj(o)
        return o


# ============================================================================
# RetNet MLP (SwiGLU)
# ============================================================================

class RetNetMLP(nn.Module):
    """
    RetNet-style MLP with SwiGLU activation.

    Matches fla.modules.mlp.GatedMLP architecture exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int = 2,
        intermediate_size: int = None,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.intermediate_size = intermediate_size

        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.swish(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# RetNet Block
# ============================================================================

class RetNetBlock(nn.Module):
    """
    RetNet block: pre-norm + retention + pre-norm + MLP.

    Matches fla.models.retnet.modeling_retnet.RetNetBlock structure.
    """

    def __init__(
        self,
        hidden_size: int,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        num_heads: int = 10,
        num_kv_heads: int = None,
        hidden_ratio: int = 2,
        intermediate_size: int = None,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps, learnable_weight=True, dim=-1)
        self.attn = MultiScaleRetention(
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            elementwise_affine=elementwise_affine,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )

        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps, learnable_weight=True, dim=-1)
        self.mlp = RetNetMLP(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cache_params=cache_params,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# RetNet Cache for autoregressive generation
# ============================================================================

class RetNetCache:
    """
    Cache for RetNet recurrent states during autoregressive generation.

    Stores per-layer:
    - recurrent_state: (B, H, K, V) - the retention hidden state matrix
    - seqlen_offset: int - current sequence length offset for rotary embeddings
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.recurrent_states = [None] * num_layers
        self.seqlen_offsets = [None] * num_layers

    def update_recurrent_state(self, layer_idx: int, new_state: torch.Tensor) -> None:
        self.recurrent_states[layer_idx] = new_state

    def update_seqlen_offset(self, layer_idx: int, offset: int) -> None:
        self.seqlen_offsets[layer_idx] = offset

    def reset(self):
        for i in range(self.num_layers):
            self.recurrent_states[i] = None
            self.seqlen_offsets[i] = None


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    RetNet (Retentive Network) language model.

    Matches fla.models.retnet architecture (RetNetForCausalLM) exactly:
    - model.embeddings -> self.embeddings
    - model.layers -> self.layers
    - model.norm -> self.norm
    - lm_head -> self.lm_head

    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm
    - Linear from level1/matmul/10_Linear
    - Swish from level1/activations/7_Swish
    - Embedding from level1/embeddings/2_Embedding
    - RetentionRecurrence from level1/ssm/11_Retention
    - SimpleRotaryEmbedding (inline, matching fla's RotaryEmbedding)

    Does not depend on flash-linear-attention at runtime.
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_hidden_layers: int = 32,
        vocab_size: int = 32000,
        num_heads: int = 10,
        num_kv_heads: int = None,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        hidden_ratio: int = 2,
        intermediate_size: int = None,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        pad_token_id: int = None,
        tie_word_embeddings: bool = False,
        # Accept extra kwargs from HF config
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Compute derived sizes
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Embedding
        self.embeddings = Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Decoder layers
        self.layers = nn.ModuleList([
            RetNetBlock(
                hidden_size=hidden_size,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_ratio=hidden_ratio,
                intermediate_size=intermediate_size,
                elementwise_affine=elementwise_affine,
                norm_eps=norm_eps,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(hidden_size, eps=norm_eps, learnable_weight=True, dim=-1)

        # LM head
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.embeddings.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_params: Optional[RetNetCache] = None,
        use_cache: bool = False,
        num_last_tokens: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            cache_params: Optional RetNetCache for autoregressive generation.
            use_cache: Whether to output/save the final recurrent state.
            num_last_tokens: If > 0, only project the last N hidden states
                through the LM head (avoids a large matmul during prefill).
                If 0, project all positions.

        Returns:
            logits: (batch_size, seq_len or num_last_tokens, vocab_size)
        """
        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cache_params=cache_params,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        logits = self.lm_head(hidden_states)
        return logits

    def _create_cache(self, batch_size: int, **kwargs) -> RetNetCache:
        """Create an empty RetNetCache for autoregressive generation."""
        return RetNetCache(num_layers=self.num_hidden_layers)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        return_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Autoregressive generation with O(1) per-step cost using cached
        recurrent states.
        """
        batch_size, prompt_len = input_ids.shape

        cache_params = self._create_cache(batch_size)

        # Prefill: process all prompt tokens, populating cache.
        # Only project the last hidden state through lm_head unless we
        # need all logits for the caller.
        prefill_logits = self.forward(
            input_ids,
            cache_params=cache_params,
            use_cache=True,
            num_last_tokens=0 if return_logits else 1,
        )

        if max_new_tokens == 0:
            if return_logits:
                return input_ids, [prefill_logits]
            return input_ids

        all_logits = [prefill_logits] if return_logits else None
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [input_ids, next_token]

        # Decode loop
        for step in range(max_new_tokens - 1):
            decode_logits = self.forward(
                next_token,
                cache_params=cache_params,
                use_cache=True,
            )
            if return_logits:
                all_logits.append(decode_logits)
            next_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        generated_ids = torch.cat(generated, dim=1)
        if return_logits:
            return generated_ids, all_logits
        return generated_ids


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 4
sequence_length = 1024
hidden_size = 2560
num_hidden_layers = 8
vocab_size = 32000
num_heads = 10


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'hidden_size': hidden_size,
        'num_hidden_layers': num_hidden_layers,
        'vocab_size': vocab_size,
        'num_heads': num_heads,
        'expand_k': 1.0,
        'expand_v': 2.0,
        'hidden_ratio': 2,
    }]
