"""
Mamba-1 State Space Model

Implements the Mamba-1 architecture aligned with HuggingFace transformers,
supporting both generic Mamba-1 (state-spaces/mamba-*) and Falcon Mamba
(tiiuae/falcon-mamba-7b) through a single unified implementation.

The only difference between the two variants is that Falcon Mamba applies
RMS normalization (without learnable weights) to the B, C, and dt SSM
parameters before discretization. This is controlled by the
`use_mixer_rms_norm` flag.

Architecture:
- Mamba-1 selective scan with input-dependent state transitions
- Gated MLP-style input projection (hidden_states + gate)
- Causal depthwise Conv1d for local context
- Optional RMS normalization of B, C, and dt (Falcon-Mamba variant)
- Sequential recurrence (no chunked parallel scan like Mamba-2)

Key config parameters for falcon-mamba-7b:
  hidden_size=4096, intermediate_size=8192, state_size=16,
  conv_kernel=4, expand=2, num_hidden_layers=64, vocab_size=65024,
  time_step_rank=256, mixer_rms_eps=1e-6, use_mixer_rms_norm=True

Key config parameters for state-spaces/mamba-2.8b:
  hidden_size=2560, intermediate_size=5120, state_size=16,
  conv_kernel=4, expand=2, num_hidden_layers=64, vocab_size=50280,
  time_step_rank=160, use_mixer_rms_norm=False

HF weight layout (prefix: backbone.):
  backbone.embeddings.weight
  backbone.layers[i].norm.weight
  backbone.layers[i].mixer.in_proj.weight
  backbone.layers[i].mixer.conv1d.weight / .bias
  backbone.layers[i].mixer.x_proj.weight
  backbone.layers[i].mixer.dt_proj.weight / .bias
  backbone.layers[i].mixer.A_log
  backbone.layers[i].mixer.D
  backbone.layers[i].mixer.out_proj.weight
  backbone.norm_f.weight
  lm_head.weight

This model uses level1 operators from KernelBench.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNormOp
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._11_Softplus import Model as Softplus
from ..level1.matmul._10_Linear import Model as Linear

# Import level1 SSM operators
from ..level1.ssm._1_MambaCausalConv1d import Model as MambaCausalConv1d
from ..level1.ssm._2_MambaCausalConv1dStep import Model as MambaCausalConv1dStep
from ..level1.ssm._5_Mamba1SelectiveScan import Model as Mamba1SelectiveScan
from ..level1.ssm._6_Mamba1SSMStep import Model as Mamba1SSMStep


# ============================================================================
# Model Variants - configs loaded from HuggingFace
# ============================================================================

VARIANTS: Dict[str, str] = {
    "falcon-mamba-7b": "tiiuae/falcon-mamba-7b",
    "mamba-2.8b": "state-spaces/mamba-2.8b-slimpj",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
    "mamba-790m": "state-spaces/mamba-790m-hf",
    "mamba-370m": "state-spaces/mamba-370m-hf",
    "mamba-130m": "state-spaces/mamba-130m-hf",
}


# ============================================================================
# Mamba1 Cache for autoregressive generation
# ============================================================================

class Mamba1Cache:
    """Cache for Mamba-1 conv and SSM states during generation."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        intermediate_size: int,
        conv_kernel_size: int,
        state_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        self.conv_kernel_size = conv_kernel_size

        # Conv states: list of (batch, intermediate_size, conv_kernel_size) per layer
        self.conv_states = []
        self.ssm_states = []
        device = torch.device(device) if device is not None else None
        for _ in range(num_layers):
            conv_state = torch.zeros(
                batch_size, intermediate_size, conv_kernel_size,
                device=device, dtype=dtype,
            )
            ssm_state = torch.zeros(
                batch_size, intermediate_size, state_size,
                device=device, dtype=dtype,
            )
            self.conv_states.append(conv_state)
            self.ssm_states.append(ssm_state)

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        """Update conv state, matching HF MambaCache/FalconMambaCache behavior."""
        conv_state = self.conv_states[layer_idx]

        # Ensure same device
        if conv_state.device != new_conv_state.device:
            conv_state = conv_state.to(new_conv_state.device)
            self.conv_states[layer_idx] = conv_state

        cache_position_clamped = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position_clamped] = new_conv_state.to(
            device=conv_state.device, dtype=conv_state.dtype
        )
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx].zero_()
        self.ssm_states[layer_idx] += new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            self.conv_states[layer_idx].zero_()
            self.ssm_states[layer_idx].zero_()


# ============================================================================
# Component Modules (using level1 operators)
# ============================================================================

class Mamba1Mixer(nn.Module):
    """
    Mamba-1 mixer block.

    Supports both generic Mamba-1 and Falcon Mamba variants.
    When use_mixer_rms_norm=True (Falcon Mamba), applies RMS normalization
    to B, C, and dt SSM parameters before discretization.

    Uses level1 operators:
    - MambaCausalConv1d for prefill-path convolution
    - MambaCausalConv1dStep for decode-path cached convolution
    - Mamba1SelectiveScan for prefill-path sequential scan
    - Mamba1SSMStep for decode-path single-step SSM update
    - RMSNorm for optional mixer parameter normalization
    - Linear for projections
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_rank: int = 256,
        use_mixer_rms_norm: bool = False,
        mixer_rms_eps: float = 1e-6,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.layer_idx = layer_idx
        self.use_conv_bias = use_conv_bias
        self.time_step_rank = time_step_rank
        self.use_mixer_rms_norm = use_mixer_rms_norm

        if intermediate_size is None:
            self.intermediate_size = int(expand * hidden_size)
        else:
            self.intermediate_size = intermediate_size

        # Activation
        self.act = Swish()

        # Causal depthwise Conv1d using level1 SSM operator (prefill path)
        # apply_silu=False because we apply SiLU manually via self.act
        self.conv1d = MambaCausalConv1d(
            d_inner=self.intermediate_size,
            kernel_size=conv_kernel,
            bias=use_conv_bias,
            apply_silu=False,
        )

        # Cached conv1d step operator (decode path, no owned parameters)
        self.conv1d_step = MambaCausalConv1dStep(apply_silu=False)

        # Input projection: hidden_size -> 2 * intermediate_size (hidden_states + gate)
        self.in_proj = Linear(hidden_size, self.intermediate_size * 2, bias=use_bias)

        # Selective projection: intermediate_size -> time_step_rank + 2*state_size
        self.x_proj = Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)

        # Time step projection: time_step_rank -> intermediate_size
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        # Skip connection
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

        # Output projection
        self.out_proj = Linear(self.intermediate_size, hidden_size, bias=use_bias)

        # Level1 SSM operators
        self.selective_scan = Mamba1SelectiveScan()
        self.ssm_step = Mamba1SSMStep()

        # Softplus for dt projection
        self.softplus_op = Softplus()

        # Optional RMSNorm for B, C, dt normalization (Falcon-Mamba variant)
        if self.use_mixer_rms_norm:
            self.b_rms_norm = RMSNormOp(self.ssm_state_size, mixer_rms_eps, learnable_weight=False, dim=-1)
            self.c_rms_norm = RMSNormOp(self.ssm_state_size, mixer_rms_eps, learnable_weight=False, dim=-1)
            self.dt_rms_norm = RMSNormOp(self.time_step_rank, mixer_rms_eps, learnable_weight=False, dim=-1)

    def forward(
        self,
        input_states: torch.Tensor,
        cache_params: Optional[Mamba1Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Mamba-1 mixer forward pass (matches HF slow_forward).
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)
        # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)

            # Check if prefill or decode
            if cache_position is not None and cache_position.shape[0] == self.conv_kernel_size:
                # Prefill path
                conv_state = F.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                )
                cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)

                # MambaCausalConv1d expects (batch, seq_len, d_inner), but hidden_states
                # is (batch, d_inner, seq_len), so we transpose, call, transpose back
                hidden_states = self.act(
                    self.conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
                )  # [batch, intermediate_size, seq_len]
            else:
                # Decode path: single-step cached convolution
                conv_state = cache_params.update_conv_state(
                    self.layer_idx, hidden_states, cache_position
                )
                conv_state = conv_state.to(self.conv1d.conv1d.weight.device)
                hidden_states = torch.sum(
                    conv_state * self.conv1d.conv1d.weight[:, 0, :], dim=-1
                )
                if self.use_conv_bias:
                    hidden_states += self.conv1d.conv1d.bias
                hidden_states = (
                    self.act(hidden_states).to(dtype).unsqueeze(-1)
                )  # [batch, intermediate_size, 1] : decoding
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device,
                dtype=dtype,
            )
            # Prefill without cache
            hidden_states = self.act(
                self.conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
            )  # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection: project to dt, B, C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        # Optional RMS normalization of B, C, dt (Falcon-Mamba variant)
        if self.use_mixer_rms_norm:
            B = self.b_rms_norm(B)
            C = self.c_rms_norm(C)
            time_step = self.dt_rms_norm(time_step)

        # dt projection + softplus
        discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
        discrete_time_step = F.softplus(discrete_time_step).transpose(
            1, 2
        )  # [batch, intermediate_size, seq_len]

        # 3.b. Discretization
        A = -torch.exp(self.A_log.float())  # [intermediate_size, state_size]
        discrete_A = torch.exp(
            A[None, :, None, :] * discrete_time_step[:, :, :, None]
        )  # [batch, intermediate_size, seq_len, state_size]
        discrete_B = (
            discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        )  # [batch, intermediate_size, seq_len, state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c. Selective scan recurrence
        scan_output, ssm_state = self.selective_scan(
            hidden_states, discrete_A, deltaB_u, C, self.D, gate, ssm_state
        )

        if cache_params is not None:
            cache_params.update_ssm_state(self.layer_idx, ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(
            scan_output.transpose(1, 2)
        )  # [batch, seq_len, hidden_size]
        return contextualized_states


class Mamba1Block(nn.Module):
    """Mamba-1 block: RMSNorm + Mamba1Mixer + residual."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_rank: int = 256,
        use_mixer_rms_norm: bool = False,
        mixer_rms_eps: float = 1e-6,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # Layer norm (level1 RMSNorm with learnable weight)
        self.norm = RMSNormOp(
            hidden_size, layer_norm_epsilon, learnable_weight=True, dim=-1
        )
        self.mixer = Mamba1Mixer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_size=state_size,
            conv_kernel=conv_kernel,
            expand=expand,
            use_bias=use_bias,
            use_conv_bias=use_conv_bias,
            time_step_rank=time_step_rank,
            use_mixer_rms_norm=use_mixer_rms_norm,
            mixer_rms_eps=mixer_rms_eps,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Mamba1Cache] = None,
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
    Mamba-1 state space model aligned with HuggingFace MambaForCausalLM
    and FalconMambaForCausalLM.

    Supports both generic Mamba-1 (state-spaces/mamba-*) and Falcon Mamba
    (tiiuae/falcon-mamba-7b) through the `use_mixer_rms_norm` flag.

    Uses level1 operators from KernelBench:
    - RMSNorm from level1/normalization/4_RMSNorm (learnable weight, dim=-1)
    - Swish/SiLU from level1/activations/7_Swish
    - Softplus from level1/activations/11_Softplus
    - Linear from level1/matmul/10_Linear
    - MambaCausalConv1d from level1/ssm/1_MambaCausalConv1d (prefill conv)
    - MambaCausalConv1dStep from level1/ssm/2_MambaCausalConv1dStep (decode conv)
    - Mamba1SelectiveScan from level1/ssm/5_Mamba1SelectiveScan (prefill scan)
    - Mamba1SSMStep from level1/ssm/6_Mamba1SSMStep (decode SSM step)

    HF model structure:
      backbone.embeddings -> self.embeddings
      backbone.layers[i]  -> self.layers[i]
      backbone.norm_f     -> self.norm_f
      lm_head             -> self.lm_head
    """

    def __init__(
        self,
        vocab_size: int = 50280,
        hidden_size: int = 2560,
        num_hidden_layers: int = 64,
        intermediate_size: int = None,
        state_size: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        time_step_rank: int = 160,
        use_mixer_rms_norm: bool = False,
        mixer_rms_eps: float = 1e-6,
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
        self.state_size = state_size
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.time_step_rank = time_step_rank
        self.use_mixer_rms_norm = use_mixer_rms_norm
        self.mixer_rms_eps = mixer_rms_eps
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.tie_word_embeddings = tie_word_embeddings

        # Compute derived sizes
        if intermediate_size is None:
            self.intermediate_size = int(expand * hidden_size)
        else:
            self.intermediate_size = intermediate_size

        # Embedding
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            Mamba1Block(
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size,
                state_size=state_size,
                conv_kernel=conv_kernel,
                expand=expand,
                use_bias=use_bias,
                use_conv_bias=use_conv_bias,
                time_step_rank=time_step_rank,
                use_mixer_rms_norm=use_mixer_rms_norm,
                mixer_rms_eps=mixer_rms_eps,
                layer_norm_epsilon=layer_norm_epsilon,
                residual_in_fp32=residual_in_fp32,
                layer_idx=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final norm
        self.norm_f = RMSNormOp(
            hidden_size, layer_norm_epsilon, learnable_weight=True, dim=-1
        )

        # LM head
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

        # Optionally tie weights
        if tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_params: Optional[Mamba1Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            cache_params: Optional Mamba1Cache for autoregressive generation
            cache_position: Position in cache (for decode steps)
            use_cache: Whether to use cache

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = Mamba1Cache(
                num_layers=self.num_hidden_layers,
                batch_size=input_ids.shape[0],
                intermediate_size=self.intermediate_size,
                conv_kernel_size=self.conv_kernel,
                state_size=self.state_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            cache_position = torch.arange(
                0, self.conv_kernel, device=input_ids.device
            )

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
        Autoregressive generation for Mamba-1.

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
        cache_params = Mamba1Cache(
            num_layers=self.num_hidden_layers,
            batch_size=batch_size,
            intermediate_size=self.intermediate_size,
            conv_kernel_size=self.conv_kernel,
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
