"""
RWKV-6 Linear Attention Model

Implements RWKV-6 architecture following the flash-linear-attention (fla) library:
- Linear-complexity receptance-weighted key-value attention
- Time mixing with data-dependent lerp and low-rank projections
- Channel mixing with squared ReLU activation
- Compatible with fla-hub/rwkv6-* HuggingFace checkpoints

Variants:
- RWKV-6-1.6B: 24 layers, hidden_size=2048
- RWKV-6-3B: 32 layers, hidden_size=2560
- RWKV-6-7B: 32 layers, hidden_size=4096

HF weight layout (prefix: model.):
  model.embeddings.weight
  model.layers[i].attn_norm.ln.weight / .ln.bias
  model.layers[i].attn.x_proj.* / .x_proj_up.* / .x_bias
  model.layers[i].attn.{r,w,k,v,g}_proj.*
  model.layers[i].attn.bonus
  model.layers[i].attn.g_norm.weight / .bias
  model.layers[i].attn.o_proj.weight
  model.layers[i].ffn_norm.ln.weight / .ln.bias
  model.layers[i].ffn.key.* / .value.* / .receptance.*
  model.norm.ln.weight / .ln.bias
  lm_head.weight

This model uses level1 operators from KernelBench.
It does NOT depend on flash-linear-attention at runtime.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.normalization._3_GroupNorm import Model as GroupNorm
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._3_Sigmoid import Model as Sigmoid
from ..level1.activations._1_ReLU import Model as ReLU
from ..level1.activations._4_Tanh import Model as Tanh
from ..level1.embeddings._2_Embedding import Model as Embedding

# Import level1 RWKV-6 SSM operators
from ..level1.ssm._7_RWKV6NaiveRecurrent import Model as RWKV6NaiveRecurrent
from ..level1.ssm._8_RWKV6TokenShift import Model as RWKV6TokenShift
from ..level1.ssm._9_RWKV6LerpLinear import Model as RWKV6LerpLinear


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "1.6B": "fla-hub/rwkv6-1.6B-finch",
    "3B": "fla-hub/rwkv6-3B-finch",
    "7B": "fla-hub/rwkv6-7B-finch",
}


# ============================================================================
# RWKV6 Cache for autoregressive generation
# ============================================================================

class RWKV6Cache:
    """
    Cache for RWKV-6 recurrent states during autoregressive generation.

    Stores per-layer:
    - recurrent_state: (B, H, K, V) - the WKV hidden state matrix
    - attn_conv_state: (B, hidden_size) - last hidden state for attention token shift
    - ffn_state: (B, hidden_size) - last hidden state for FFN token shift
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        self.num_layers = num_layers
        # Recurrent states for each layer: (B, H, K, V)
        self.recurrent_states = [None] * num_layers
        # Token shift states for attention and FFN: (B, hidden_size)
        self.attn_conv_states = [None] * num_layers
        self.ffn_states = [None] * num_layers

    def update_recurrent_state(
        self, layer_idx: int, new_state: torch.Tensor
    ) -> None:
        self.recurrent_states[layer_idx] = new_state

    def update_attn_conv_state(
        self, layer_idx: int, new_state: torch.Tensor
    ) -> None:
        self.attn_conv_states[layer_idx] = new_state

    def update_ffn_state(
        self, layer_idx: int, new_state: torch.Tensor
    ) -> None:
        self.ffn_states[layer_idx] = new_state

    def reset(self):
        for i in range(self.num_layers):
            self.recurrent_states[i] = None
            self.attn_conv_states[i] = None
            self.ffn_states[i] = None


# ============================================================================
# RWKV6 Attention (Time Mixing)
# ============================================================================

class RWKV6Attention(nn.Module):
    """
    RWKV-6 attention (time mixing) block.

    Matches fla.layers.rwkv6.RWKV6Attention architecture exactly.
    Uses low-rank projections for modulation vectors and data-dependent
    lerp mixing for r, w, k, v, g projections.

    Uses level1 operators:
    - RWKV6LerpLinear for x_proj lerp mixing (learnable mu)
    - RWKV6LerpLinear for data-dependent r, w, k, v, g projections (data-dependent mu)
    - RWKV6NaiveRecurrent for linear attention recurrence
    - RWKV6TokenShift for token shift delta
    - GroupNorm for per-position normalisation
    - Linear for output projection
    - Swish for gating activation
    - Tanh for x_proj activation
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)

        assert self.key_dim % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Level1 operators for token shift and activations
        self.token_shift = RWKV6TokenShift()
        self.tanh = Tanh()
        self.swish = Swish()

        # x_proj: LerpLinear -> tanh -> Linear to produce 5 modulation vectors
        self.x_proj = RWKV6LerpLinear(hidden_size, proj_low_rank_dim * 5)
        self.x_proj_up = Linear(proj_low_rank_dim * 5, hidden_size, bias=False)
        self.x_bias = nn.Parameter(torch.zeros(5, hidden_size))

        # Data-dependent lerp projections for r, w, k, v, g (level1 operators)
        self.r_proj = RWKV6LerpLinear(hidden_size, self.key_dim, learnable_mu=False)
        self.w_proj = RWKV6LerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim, learnable_mu=False)
        self.k_proj = RWKV6LerpLinear(hidden_size, self.key_dim, learnable_mu=False)
        self.v_proj = RWKV6LerpLinear(hidden_size, self.value_dim, learnable_mu=False)
        self.g_proj = RWKV6LerpLinear(hidden_size, self.value_dim, learnable_mu=False)

        # Bonus parameter for current-token contribution
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_k_dim))

        # Group norm and output projection (level1 operators)
        # fla's GroupNorm normalizes per-position (each token independently),
        # treating the hidden dim as (num_heads, head_dim) groups.
        # nn.GroupNorm expects (N, C, *) so we reshape (B, T, C) -> (B*T, C, 1)
        self.g_norm = GroupNorm(self.value_dim, self.num_heads)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        # Level1 RWKV6 naive recurrent operator
        self.rwkv6_recurrent = RWKV6NaiveRecurrent(scale=1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[RWKV6Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute token-shift delta (level1 operator)
        # prev_hidden is the last hidden state from a previous forward pass (if cached)
        prev_hidden = (
            cache_params.attn_conv_states[self.layer_idx]
            if cache_params is not None and cache_params.attn_conv_states[self.layer_idx] is not None
            else None
        )
        delta = self.token_shift(hidden_states, prev_hidden=prev_hidden)

        # Save last hidden state for future token shifts
        if cache_params is not None:
            cache_params.update_attn_conv_state(self.layer_idx, hidden_states[:, -1])

        # Low-rank projection: produce 5 modulation vectors
        x = self.x_proj(hidden_states, delta).view(
            batch_size, seq_len, -1, self.proj_low_rank_dim
        )
        x = torch.einsum(
            'b t n r, h n r -> b t n h',
            self.tanh(x),
            self.x_proj_up.weight.view(hidden_size, 5, -1),
        )

        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)

        # Data-dependent lerp projections (level1 operators)
        r = self.r_proj(hidden_states, delta, mu=r)
        w = self.w_proj(hidden_states, delta, mu=w)
        k = self.k_proj(hidden_states, delta, mu=k)
        v = self.v_proj(hidden_states, delta, mu=v)
        g = self.g_proj(hidden_states, delta, mu=g)

        # Reshape for multi-head: (B, T, H, D)
        r = r.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        w = w.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # Negate and exponentiate for decay
        w = -torch.exp(w)

        # Transpose to (B, H, T, D) for recurrent computation
        r = r.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        w = w.transpose(1, 2)

        # Get initial recurrent state from cache (if available)
        initial_state = None
        if cache_params is not None:
            initial_state = cache_params.recurrent_states[self.layer_idx]

        # Run RWKV6 linear attention (level1 operator)
        # Note: In fla's reference code, q=receptance, so we pass r as the first arg
        o, final_state = self.rwkv6_recurrent(
            q=r, k=k, v=v, w=w,
            u=self.bonus,
            initial_state=initial_state,
            output_final_state=use_cache,
        )

        # Save recurrent state for future decode steps
        if cache_params is not None and final_state is not None:
            cache_params.update_recurrent_state(self.layer_idx, final_state)

        # Transpose back to (B, T, H*D) and apply group norm + gate
        o = o.transpose(1, 2).contiguous()
        o = o.view(batch_size, seq_len, -1)

        # Apply GroupNorm per-position: reshape (B, T, C) -> (B*T, C, 1)
        # for nn.GroupNorm, then back to (B, T, C).
        # This normalizes each group of channels independently at each position,
        # matching fla's GroupNorm behavior.
        B_T = batch_size * seq_len
        o = self.g_norm(o.reshape(B_T, -1, 1)).reshape(batch_size, seq_len, -1)

        # Apply swish gate (level1 operator)
        o = o * self.swish(g)

        # Output projection (level1 operator)
        o = self.o_proj(o)
        return o


# ============================================================================
# RWKV6 Feed-Forward (Channel Mixing)
# ============================================================================

class RWKV6FeedForward(nn.Module):
    """
    RWKV-6 Channel Mixing (FFN) block.

    Matches fla.models.rwkv6.modeling_rwkv6.RWKV6FeedForward architecture.
    Uses LerpLinear for key and receptance projections with squared ReLU.

    Uses level1 operators:
    - RWKV6LerpLinear for key and receptance projections (learnable mu)
    - RWKV6TokenShift for token shift delta
    - Linear for value projection
    - ReLU for activation (applied as squared ReLU)
    - Sigmoid for gating
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: float = 3.5,
        intermediate_size: int = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.intermediate_size = intermediate_size

        # Level1 operators
        self.token_shift = RWKV6TokenShift()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        # LerpLinear projections (level1 operators)
        self.key = RWKV6LerpLinear(hidden_size, intermediate_size)
        self.value = Linear(intermediate_size, hidden_size, bias=False)
        self.receptance = RWKV6LerpLinear(hidden_size, hidden_size)

        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        cache_params: Optional[RWKV6Cache] = None,
    ) -> torch.Tensor:
        # Compute token-shift delta (level1 operator)
        # prev_hidden is the last hidden state from a previous forward pass (if cached)
        prev_hidden = (
            cache_params.ffn_states[self.layer_idx]
            if cache_params is not None and cache_params.ffn_states[self.layer_idx] is not None
            else None
        )
        delta = self.token_shift(x, prev_hidden=prev_hidden)

        # Save last hidden state for future token shifts
        if cache_params is not None:
            cache_params.update_ffn_state(self.layer_idx, x[:, -1])

        # Squared ReLU activation (level1 ReLU, then square)
        key = self.key(x, delta)
        key = self.relu(key.float()).square().to(key.dtype)
        value = self.value(key)
        receptance = self.receptance(x, delta)
        return self.sigmoid(receptance) * value


# ============================================================================
# RWKV6 Block
# ============================================================================

class RWKV6Block(nn.Module):
    """
    RWKV-6 block: pre-norm + attention + pre-norm + FFN.

    Matches fla.models.rwkv6.modeling_rwkv6.RWKV6Block structure.
    The first block has an extra pre_norm (when norm_first=True).

    Uses level1 operators:
    - LayerNorm for normalisation
    """

    def __init__(
        self,
        hidden_size: int,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 64,
        proj_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        hidden_ratio: float = 3.5,
        intermediate_size: int = None,
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        norm_first: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        # Pre-norm for the very first block (before the first residual)
        if norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(hidden_size)
            # Copy bias setting
            if not norm_bias:
                self.pre_norm.ln.bias = None

        self.attn_norm = LayerNorm(hidden_size)
        self.attn = RWKV6Attention(
            hidden_size=hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )

        self.ffn_norm = LayerNorm(hidden_size)
        self.ffn = RWKV6FeedForward(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            intermediate_size=intermediate_size,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[RWKV6Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states

        hidden_states = self.attn_norm(residual)
        hidden_states = self.attn(
            hidden_states,
            cache_params=cache_params,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    RWKV-6 linear attention language model.

    Matches fla.models.rwkv6 architecture (RWKV6ForCausalLM) exactly:
    - model.embeddings -> self.embeddings
    - model.layers -> self.layers
    - model.norm -> self.norm
    - lm_head -> self.lm_head

    Uses level1 operators from KernelBench:
    - LayerNorm from level1/normalization/6_LayerNorm
    - GroupNorm from level1/normalization/3_GroupNorm (per-position norm)
    - Linear from level1/matmul/10_Linear
    - Swish/SiLU from level1/activations/7_Swish
    - Sigmoid from level1/activations/3_Sigmoid
    - ReLU from level1/activations/1_ReLU
    - Tanh from level1/activations/4_Tanh
    - Embedding from level1/embeddings/2_Embedding
    - RWKV6NaiveRecurrent from level1/ssm/7_RWKV6NaiveRecurrent
    - RWKV6TokenShift from level1/ssm/8_RWKV6TokenShift
    - RWKV6LerpLinear from level1/ssm/9_RWKV6LerpLinear (learnable & data-dependent mu)

    Does not depend on flash-linear-attention at runtime.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        vocab_size: int = 65536,
        num_heads: int = 64,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        proj_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        hidden_ratio: float = 3.5,
        intermediate_size: int = None,
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        norm_first: bool = True,
        pad_token_id: int = None,
        tie_word_embeddings: bool = False,
        # Accept extra kwargs from HF config
        **kwargs,
    ):
        super().__init__()

        # Store all config parameters (following Mamba2/Llama pattern)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.norm_first = norm_first
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # Compute derived sizes
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Embedding (level1 operator)
        self.embeddings = Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Decoder layers
        self.layers = nn.ModuleList([
            RWKV6Block(
                hidden_size=hidden_size,
                expand_k=expand_k,
                expand_v=expand_v,
                num_heads=num_heads,
                proj_low_rank_dim=proj_low_rank_dim,
                gate_low_rank_dim=gate_low_rank_dim,
                hidden_ratio=hidden_ratio,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                norm_bias=norm_bias,
                norm_first=norm_first,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final norm (level1 operator)
        self.norm = LayerNorm(hidden_size)

        # LM head (level1 operator)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.embeddings.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_params: Optional[RWKV6Cache] = None,
        use_cache: bool = False,
        num_last_tokens: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            cache_params: Optional RWKV6Cache for autoregressive generation.
                When provided, recurrent and token-shift states are read from
                and written to the cache.
            use_cache: Whether to output/save the final recurrent state.
                Must be True during generation so that the recurrent state
                is stored in cache_params for subsequent decode steps.
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

    def _create_cache(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ) -> RWKV6Cache:
        """Create an empty RWKV6Cache for autoregressive generation."""
        return RWKV6Cache(
            num_layers=self.num_hidden_layers,
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            dtype=dtype,
            device=device,
        )

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

        The generation proceeds in two phases:
        1. Prefill: process all prompt tokens at once, populating the cache
           with recurrent states and token-shift states.
        2. Decode: generate one token at a time, reusing cached states so
           each step processes only a single token.

        Args:
            input_ids: (batch_size, prompt_len) prompt token IDs
            max_new_tokens: Maximum number of new tokens to generate.
                If 0, only prefill and return logits.
            return_logits: If True, also return logits at each step.
            **kwargs: Ignored (for API compatibility).

        Returns:
            If return_logits=False:
                generated_ids: (batch_size, prompt_len + max_new_tokens)
            If return_logits=True:
                Tuple of (generated_ids, logits_list)
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Create cache
        cache_params = self._create_cache(
            batch_size=batch_size,
            dtype=self.embeddings.embedding.weight.dtype,
            device=device,
        )

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

        # Decode loop: one token at a time, O(1) per step
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
hidden_size = 2048
num_hidden_layers = 8
vocab_size = 65536
num_heads = 32


def get_inputs():
    return [torch.randint(0, vocab_size, (batch_size, sequence_length))]


def get_init_inputs():
    return [{
        'hidden_size': hidden_size,
        'num_hidden_layers': num_hidden_layers,
        'vocab_size': vocab_size,
        'num_heads': num_heads,
        'expand_k': 1.0,
        'expand_v': 1.0,
        'proj_low_rank_dim': 64,
        'gate_low_rank_dim': 128,
        'hidden_ratio': 3.5,
    }]
