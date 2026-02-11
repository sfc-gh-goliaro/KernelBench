"""
Mamba-2 State Space Model (Mamba-Codestral)

Implements Mamba-2 with State-Space Duality (SSD) aligned with the HuggingFace
transformers Mamba2ForCausalLM architecture (mistralai/Mamba-Codestral-7B-v0.1):
- Selective scan mechanism with chunked parallel computation
- Multi-head structure for state-space computation
- Gated RMSNorm output normalization
- Conv1d for local context

Key config parameters for Mamba-Codestral-7B-v0.1:
  hidden_size=4096, num_heads=128, head_dim=64, state_size=128,
  expand=2, conv_kernel=4, n_groups=8, num_hidden_layers=64,
  chunk_size=256, vocab_size=32768

HF weight layout (prefix: backbone.):
  backbone.embeddings.weight
  backbone.layers[i].norm.weight
  backbone.layers[i].mixer.in_proj.weight
  backbone.layers[i].mixer.conv1d.weight / .bias
  backbone.layers[i].mixer.dt_bias
  backbone.layers[i].mixer.A_log
  backbone.layers[i].mixer.D
  backbone.layers[i].mixer.norm.weight
  backbone.layers[i].mixer.out_proj.weight
  backbone.norm_f.weight
  lm_head.weight

This model uses level1 operators from KernelBench.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNormOp
from ..level1.normalization._7_RMSNormGated import Model as RMSNormGated
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._11_Softplus import Model as Softplus
from ..level1.matmul._10_Linear import Model as Linear

# Import level1 SSM operators
from ..level1.ssm._1_MambaCausalConv1d import Model as MambaCausalConv1d
from ..level1.ssm._2_MambaCausalConv1dStep import Model as MambaCausalConv1dStep
from ..level1.ssm._3_Mamba2SSDChunkedScan import Model as Mamba2SSDChunkedScan
from ..level1.ssm._4_Mamba2StateUpdateStep import Model as Mamba2StateUpdateStep


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Codestral-7B": "mistralai/Mamba-Codestral-7B-v0.1",
}


# ============================================================================
# Mamba2 Cache for autoregressive generation
# ============================================================================

class Mamba2Cache:
    """Cache for Mamba2 conv and SSM states during generation."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        conv_dim: int,
        conv_kernel_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        self.conv_states = torch.zeros(
            num_layers, batch_size, conv_dim, conv_kernel_size,
            device=device, dtype=dtype,
        )
        self.ssm_states = torch.zeros(
            num_layers, batch_size, num_heads, head_dim, state_size,
            device=device, dtype=dtype,
        )

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False
    ) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                self.conv_states.device
            )
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class Mamba2Mixer(nn.Module):
    """
    Mamba-2 mixer with State-Space Duality (SSD).

    Matches HuggingFace Mamba2Mixer torch_forward path exactly.
    Uses level1 operators:
    - MambaCausalConv1d for prefill-path convolution
    - MambaCausalConv1dStep for decode-path cached convolution
    - Mamba2SSDChunkedScan for prefill-path chunked parallel SSD
    - Mamba2StateUpdateStep for decode-path single-step SSM update
    - Linear, Softplus, RMSNorm for projections and activations
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 128,
        head_dim: int = 64,
        state_size: int = 128,
        conv_kernel: int = 4,
        expand: int = 2,
        n_groups: int = 8,
        chunk_size: int = 256,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_limit: tuple = (0.0, float("inf")),
        time_step_rank: int = 256,
        layer_norm_epsilon: float = 1e-5,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * hidden_size)
        self.time_step_rank = time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.layer_norm_epsilon = layer_norm_epsilon

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # Causal depthwise Conv1d using level1 SSM operator (prefill path)
        self.conv1d = MambaCausalConv1d(
            d_inner=self.conv_dim,
            kernel_size=conv_kernel,
            bias=use_conv_bias,
            apply_silu=False,  # SiLU applied via self.swish for consistency
        )

        # Cached conv1d step operator (decode path, no owned parameters)
        self.conv1d_step = MambaCausalConv1dStep(apply_silu=False)

        # Input projection: projects to gate + conv_input + dt
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = Linear(hidden_size, projection_size, bias=use_bias)

        # dt_bias parameter
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm (level1 operator, gate applied before norm = Mamba2 style)
        self.norm = RMSNormGated(self.intermediate_size, eps=layer_norm_epsilon)

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.num_heads))

        # Output projection
        self.out_proj = Linear(self.intermediate_size, hidden_size, bias=use_bias)

        # Level1 SSM operators for scan
        self.ssd_scan = Mamba2SSDChunkedScan(chunk_size=chunk_size)
        self.ssm_step = Mamba2StateUpdateStep()

        # Level1 operators for activations
        self.swish = Swish()
        self.softplus = Softplus()

    def _expand_groups(self, tensor, batch_size):
        """Expand grouped B or C from n_groups to num_heads."""
        tensor = tensor.reshape(batch_size, self.n_groups, -1)[..., None, :]
        tensor = tensor.expand(
            batch_size, self.n_groups, self.num_heads // self.n_groups, tensor.shape[-1]
        ).contiguous()
        return tensor.reshape(batch_size, -1, tensor.shape[-1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Mamba2 mixer forward pass (matches HF torch_forward).
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Input projection
        projected_states = self.in_proj(hidden_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Convolution
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # Decode step: single-step cached convolution via MambaCausalConv1dStep
            cache_params.update_conv_state(
                layer_idx=self.layer_idx,
                new_conv_state=hidden_states_B_C,
                cache_init=False,
            )
            conv_states = cache_params.conv_states[self.layer_idx].to(
                device=self.conv1d.conv1d.weight.device
            )
            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.conv1d.bias
            hidden_states_B_C = self.swish(hidden_states_B_C)
        else:
            # Prefill: full sequence convolution via MambaCausalConv1d
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = F.pad(
                    hidden_states_B_C_transposed,
                    (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                )
                cache_params.update_conv_state(
                    layer_idx=self.layer_idx,
                    new_conv_state=conv_states,
                    cache_init=True,
                )
            # MambaCausalConv1d handles transpose, conv, truncation internally
            # apply_silu=False, so we apply swish manually for consistency
            hidden_states_B_C = self.swish(
                self.conv1d(hidden_states_B_C)
            )

        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())  # [num_heads]

        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # --- Decode path: single-step SSM via Mamba2StateUpdateStep ---
            cache_device = cache_params.ssm_states.device

            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = self.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A_expanded = (
                A[..., None, None]
                .expand(self.num_heads, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dA = (torch.exp(dt[..., None] * A_expanded)).to(device=cache_device)

            # Expand B from n_groups to num_heads
            B = self._expand_groups(B, batch_size)
            dB = dt[..., None] * B[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)

            # Mamba2StateUpdateStep: h_new = dA * h + dB * x, y = C @ h_new + D * x
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)
            h_new = cache_params.ssm_states[self.layer_idx] * dA + dBx
            cache_params.update_ssm_state(
                layer_idx=self.layer_idx, new_ssm_state=h_new
            )

            # Expand C from n_groups to num_heads
            C = self._expand_groups(C, batch_size)

            # Output: y = C @ h_new + D * x
            ssm_states = cache_params.ssm_states[self.layer_idx].to(
                device=C.device, dtype=C.dtype
            )
            ssm_states_reshaped = ssm_states.view(
                batch_size * self.num_heads, self.head_dim, self.ssm_state_size
            )
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # --- Prefill path: SSD chunked scan via Mamba2SSDChunkedScan ---
            dt = self.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(
                batch_size, seq_len, -1, self.head_dim
            ).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(
                self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
            )
            C = C.repeat_interleave(
                self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
            )

            # Discretize x and A
            x_raw = hidden_states  # Save un-discretized for D residual
            hidden_states = hidden_states * dt[..., None]
            A_dt = A.to(hidden_states.dtype) * dt

            # Determine previous state for cache
            previous_state = None
            if (
                cache_params is not None
                and cache_position is not None
                and cache_position[0] > 0
            ):
                previous_state = cache_params.ssm_states[self.layer_idx]

            # Mamba2SSDChunkedScan handles: padding, chunking, segment_sum,
            # intra-chunk, inter-chunk recurrence, output computation
            y, ssm_state = self.ssd_scan(
                hidden_states, A_dt, B, C, self.D, x_raw, previous_state
            )

            # Cache SSM state
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(
                    layer_idx=self.layer_idx, new_ssm_state=ssm_state
                )

        scan_output = self.norm(y, gate)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


class Mamba2Block(nn.Module):
    """Mamba-2 block: RMSNorm + Mamba2Mixer + residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 128,
        head_dim: int = 64,
        state_size: int = 128,
        conv_kernel: int = 4,
        expand: int = 2,
        n_groups: int = 8,
        chunk_size: int = 256,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_limit: tuple = (0.0, float("inf")),
        time_step_rank: int = 256,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # Layer norm (level1 RMSNorm with learnable weight)
        self.norm = RMSNormOp(hidden_size, layer_norm_epsilon, learnable_weight=True, dim=-1)
        self.mixer = Mamba2Mixer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            conv_kernel=conv_kernel,
            expand=expand,
            n_groups=n_groups,
            chunk_size=chunk_size,
            use_bias=use_bias,
            use_conv_bias=use_conv_bias,
            time_step_limit=time_step_limit,
            time_step_rank=time_step_rank,
            layer_norm_epsilon=layer_norm_epsilon,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(
            hidden_states, cache_params=cache_params, cache_position=cache_position
        )
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    Mamba-2 state space model aligned with HuggingFace Mamba2ForCausalLM.

    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm (learnable weight, dim=-1)
    - RMSNormGated from level1/normalization/7_RMSNormGated (gated output norm)
    - Swish/SiLU from level1/activations/7_Swish
    - Softplus from level1/activations/11_Softplus
    - Linear from level1/matmul/10_Linear
    - MambaCausalConv1d from level1/ssm/1_MambaCausalConv1d (prefill conv)
    - MambaCausalConv1dStep from level1/ssm/2_MambaCausalConv1dStep (decode conv)
    - Mamba2SSDChunkedScan from level1/ssm/3_Mamba2SSDChunkedScan (prefill SSD)
    - Mamba2StateUpdateStep from level1/ssm/4_Mamba2StateUpdateStep (decode SSM)

    HF model structure:
      backbone.embeddings -> self.embeddings
      backbone.layers[i]  -> self.layers[i]
      backbone.norm_f     -> self.norm_f
      lm_head             -> self.lm_head
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 4096,
        num_hidden_layers: int = 64,
        num_heads: int = 128,
        head_dim: int = 64,
        state_size: int = 128,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 8,
        chunk_size: int = 256,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_limit: tuple = (0.0, float("inf")),
        time_step_rank: int = 256,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        tie_word_embeddings: bool = False,
        # Accept extra kwargs from HF config
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.time_step_limit = time_step_limit
        self.time_step_rank = time_step_rank
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.tie_word_embeddings = tie_word_embeddings

        # Compute derived sizes
        self.intermediate_size = int(expand * hidden_size)
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size

        # Embedding
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            Mamba2Block(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=state_size,
                conv_kernel=conv_kernel,
                expand=expand,
                n_groups=n_groups,
                chunk_size=chunk_size,
                use_bias=use_bias,
                use_conv_bias=use_conv_bias,
                time_step_limit=time_step_limit,
                time_step_rank=time_step_rank,
                layer_norm_epsilon=layer_norm_epsilon,
                residual_in_fp32=residual_in_fp32,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final norm
        self.norm_f = RMSNormOp(hidden_size, layer_norm_epsilon, learnable_weight=True, dim=-1)

        # LM head
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            cache_params: Optional Mamba2Cache for autoregressive generation
            cache_position: Position in cache (for decode steps)
            use_cache: Whether to use cache

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = Mamba2Cache(
                num_layers=self.num_hidden_layers,
                batch_size=input_ids.shape[0],
                conv_dim=self.conv_dim,
                conv_kernel_size=self.conv_kernel,
                intermediate_size=self.intermediate_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                state_size=self.state_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            cache_position = torch.arange(0, self.conv_kernel, device=input_ids.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
            )

        hidden_states = self.norm_f(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        return_logits: bool = False,
        **kwargs,  # Accept and ignore block_table etc.
    ) -> torch.Tensor:
        """
        Autoregressive generation for Mamba2 (no KV cache / block_table).

        Args:
            input_ids: (batch_size, prompt_len) prompt token IDs
            max_new_tokens: Maximum number of new tokens to generate.
                If 0, only prefill and return logits.
            return_logits: If True, also return logits at each step.
            **kwargs: Ignored (for API compatibility with attention models).

        Returns:
            If return_logits=False:
                generated_ids: (batch_size, prompt_len + max_new_tokens)
            If return_logits=True:
                Tuple of (generated_ids, logits_list)
        """
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device

        # Create cache
        cache_params = Mamba2Cache(
            num_layers=self.num_hidden_layers,
            batch_size=batch_size,
            conv_dim=self.conv_dim,
            conv_kernel_size=self.conv_kernel,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.state_size,
            dtype=self.embeddings.weight.dtype,
            device=device,
        )

        # Prefill: process all prompt tokens
        cache_position = torch.arange(0, self.conv_kernel, device=device)
        prefill_logits = self.forward(
            input_ids,
            cache_params=cache_params,
            cache_position=cache_position,
            use_cache=False,  # We already created cache
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
            cache_position = torch.tensor(
                [prompt_len + step], device=device, dtype=torch.long
            )
            decode_logits = self.forward(
                next_token,
                cache_params=cache_params,
                cache_position=cache_position,
                use_cache=False,
            )
            if return_logits:
                all_logits.append(decode_logits)
            next_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        generated_ids = torch.cat(generated, dim=1)
        if return_logits:
            return generated_ids, all_logits
        return generated_ids
