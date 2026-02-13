"""
CosyVoice3 Text-to-Speech Model (Fun-CosyVoice3-0.5B-2512)

Implements the full CosyVoice3 TTS pipeline:
1. CosyVoice3LM: KB-native Qwen2 LLM that generates speech tokens from text
2. CausalMaskedDiffWithDiT: Flow matching model that converts speech tokens to mel spectrograms
3. CausalHiFTGenerator: HiFi-GAN vocoder that converts mel spectrograms to waveforms

Architecture aligned with: https://github.com/FunAudioLLM/CosyVoice
Target model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512

This model uses level1 operators from KernelBench wherever possible.
"""

import math
import numpy as np
from scipy.signal import get_window
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

# Level1 operator imports
from ..level1.normalization._6_LayerNorm import Model as LayerNorm
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.activations._8_GELU import Model as GELUAct
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.embeddings._1_RotaryEmbedding import Model as RotaryEmbedding
from ..level1.embeddings._2_Embedding import Model as Embedding
from ..level1.regularization._1_Dropout import Model as Dropout
from ..level1.attention._2_Attention import ScaledDotProductAttention
from ..level1.upsampling._2_Interpolate import Model as Interpolate


# ============================================================================
# Variant Configuration
# ============================================================================

VARIANTS: Dict[str, str] = {
    "CosyVoice3-0.5B": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}

# Default hyperparameters for Fun-CosyVoice3-0.5B-2512
DEFAULT_CONFIG = {
    # LLM
    "speech_token_size": 6561,
    "llm_input_size": 896,
    "llm_output_size": 896,
    # Flow
    "flow_input_size": 80,
    "flow_output_size": 80,
    "flow_vocab_size": 6561,
    "flow_spk_embed_dim": 192,
    "flow_token_mel_ratio": 2,
    "flow_pre_lookahead_len": 3,
    # Pre-lookahead layer
    "pre_lookahead_in_channels": 80,
    "pre_lookahead_channels": 1024,
    # DiT
    "dit_dim": 1024,
    "dit_depth": 22,
    "dit_heads": 16,
    "dit_dim_head": 64,
    "dit_dropout": 0.0,
    "dit_ff_mult": 2,
    "dit_mel_dim": 80,
    "dit_mu_dim": 80,
    "dit_spk_dim": 80,
    "dit_static_chunk_size": 50,  # chunk_size * token_mel_ratio = 25 * 2
    "dit_num_decoding_left_chunks": -1,
    # HiFT Vocoder
    "hift_in_channels": 80,
    "hift_base_channels": 512,
    "hift_nb_harmonics": 8,
    "hift_sampling_rate": 24000,
    "hift_upsample_rates": [8, 5, 3],
    "hift_upsample_kernel_sizes": [16, 11, 7],
    "hift_istft_n_fft": 16,
    "hift_istft_hop_len": 4,
    "hift_resblock_kernel_sizes": [3, 7, 11],
    "hift_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "hift_source_resblock_kernel_sizes": [7, 7, 11],
    "hift_source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "hift_conv_pre_look_right": 4,
    # CFM
    "cfm_inference_cfg_rate": 0.7,
    "cfm_t_scheduler": "cosine",
    "cfm_n_timesteps": 10,
}


# ============================================================================
# Utility Functions
# ============================================================================

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.
    True for padded positions, False for valid positions.
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps with chunk size (streaming encoder)."""
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
):
    """Apply optional chunk mask for encoder."""
    if static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_left_chunks, xs.device
        )
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    return chunk_masks


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# ============================================================================
# 1. CosyVoice3 LLM Component
# ============================================================================

class Qwen2Attention(nn.Module):
    """Qwen2-style multi-head attention with GQA and RoPE.

    Level1 operators: Linear, RotaryEmbedding, ScaledDotProductAttention
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        # Q/K/V projections with bias (Qwen2 uses bias=True for QKV)
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        # Output projection without bias
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE (layout bhsd for compatibility with SDPA)
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_position_embeddings,
            base=rope_theta,
            layout="bhsd",
        )

        # Attention kernel
        self.sdpa = ScaledDotProductAttention(mode="sdpa")

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA."""
        if self.num_kv_groups == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_kv_groups, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_kv_groups, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections -> (B, heads, S, D)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary_emb(q, k, position_ids)

        # Append to KV cache if provided
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Expand KV heads for GQA
        k_expanded = self._repeat_kv(k)
        v_expanded = self._repeat_kv(v)

        # Attention
        attn_output = self.sdpa(q, k_expanded, v_expanded, attn_mask=attention_mask, is_causal=(past_key_value is None and attention_mask is None))

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, new_cache


class Qwen2MLP(nn.Module):
    """Qwen2 SiLU-gated MLP.

    Level1 operators: Linear, Swish
    """

    def __init__(self, hidden_size: int = 896, intermediate_size: int = 4864):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = Swish()  # SiLU = Swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Single Qwen2 decoder layer.

    Level1 operators: RMSNorm, Qwen2Attention, Qwen2MLP
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class Qwen2Model(nn.Module):
    """Qwen2 base model (decoder stack without LM head).

    Level1 operators: Embedding, RMSNorm, and sub-module operators
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass through the Qwen2 model.

        Args:
            inputs_embeds: (B, S, D) pre-embedded input
            attention_mask: Optional attention mask
            position_ids: (B, S) position indices
            past_key_values: List of (key, value) tuples per layer
            use_cache: Whether to return new KV cache

        Returns:
            (hidden_states, new_past_key_values)
        """
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, layer_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
            )
            if use_cache:
                new_past_key_values.append(layer_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past_key_values


class Qwen2Encoder(nn.Module):
    """Qwen2-based encoder for CosyVoice3 LLM component.

    Implements the same interface as the original CosyVoice Qwen2Encoder
    but uses KB level1 operators exclusively instead of HuggingFace's
    Qwen2ForCausalLM.

    The Qwen2.5-0.5B architecture:
    - 24 decoder layers with GQA (14 heads, 2 KV heads, head_dim=64)
    - SiLU-gated MLP (intermediate_size=4864)
    - RoPE (theta=1M), RMSNorm (eps=1e-6)
    - Q/K/V with bias, O/MLP without bias
    - Tied word embeddings (vocab_size=151936)

    Level1 operators: RMSNorm, Linear, Swish, Embedding, RotaryEmbedding,
                      ScaledDotProductAttention
    """

    def __init__(
        self,
        pretrain_path: str = "Qwen/Qwen2.5-0.5B",
        vocab_size: int = 151936,
        hidden_size: int = 896,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.model = Qwen2Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        """Full encoder forward pass.

        Args:
            xs: (B, T, D) input embeddings
            xs_lens: (B,) sequence lengths

        Returns:
            (hidden_states, mask) where mask is (B, 1, T)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        # Build position ids
        position_ids = torch.arange(T, device=xs.device).unsqueeze(0).expand(xs.size(0), -1)
        # Build causal + padding mask for SDPA: (B, 1, T, T)
        # Causal mask: upper triangular
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=xs.device), diagonal=1)
        # Padding mask: mask out padded positions in keys
        # Use torch.where to avoid 0 * -inf = NaN
        pad_mask = masks.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T) bool
        attn_mask = causal_mask.unsqueeze(0).expand(xs.size(0), -1, -1, -1).clone()
        attn_mask = attn_mask.masked_fill(~pad_mask, float("-inf"))

        hidden_states, _ = self.model(
            inputs_embeds=xs,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return hidden_states, masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        """Single autoregressive step with KV cache.

        Args:
            xs: (B, S, D) input embeddings (S=full seq for first call, S=1 after)
            masks: (B, S, total_S) causal attention masks
            cache: List of (key, value) tuples per layer, or None

        Returns:
            (hidden_states, new_cache)
        """
        # Determine position_ids from cache
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            past_len = cache[0][0].size(2)
        else:
            past_len = 0
            cache = None

        seq_len = xs.size(1)
        total_len = past_len + seq_len
        position_ids = torch.arange(past_len, total_len, device=xs.device).unsqueeze(0).expand(xs.size(0), -1)

        # Build attention mask for SDPA
        # For the new tokens attending to all (past + new) tokens, with causal constraint
        # masks shape from caller: (B, S, total_S) -> we need (B, 1, S, total_S)
        input_masks = masks[:, -seq_len:, :]  # (B, S, total_S) bool
        # Convert bool mask to float mask for SDPA: True=attend (0), False=masked (-inf)
        # Use torch.where to avoid 0 * -inf = NaN
        attn_mask = torch.zeros_like(input_masks, dtype=xs.dtype).unsqueeze(1)  # (B, 1, S, total_S)
        attn_mask = attn_mask.masked_fill(~input_masks.unsqueeze(1), float("-inf"))

        hidden_states, new_cache = self.model(
            inputs_embeds=xs,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return hidden_states, new_cache


class CosyVoice3LM(nn.Module):
    """
    CosyVoice3 Language Model component.

    Uses a KB-native Qwen2 model as backbone with speech token embeddings
    and decoder head. Generates speech tokens autoregressively from text input.

    Level1 operators used:
    - Embedding: speech_embedding, Qwen2 token embeddings
    - Linear: llm_decoder, Q/K/V/O projections, MLP projections
    - RMSNorm: layer normalization throughout Qwen2
    - Swish: SiLU activation in MLP
    - RotaryEmbedding: position encoding in attention
    - ScaledDotProductAttention: attention computation
    """

    def __init__(
        self,
        llm_input_size: int = 896,
        llm_output_size: int = 896,
        speech_token_size: int = 6561,
        llm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size

        # Special token IDs
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        # LLM backbone (Qwen2)
        self.llm = llm if llm is not None else Qwen2Encoder()

        # Speech token embedding -- level1 Embedding
        self.speech_embedding = Embedding(speech_token_size + 200, llm_input_size)

        # Decoder head -- level1 Linear (no bias)
        self.llm_decoder = Linear(llm_output_size, speech_token_size + 200, bias=False)

        # Stop token IDs
        self.stop_token_ids = [speech_token_size + i for i in range(200)]

    def forward(self, lm_input: torch.Tensor, lm_input_len: torch.Tensor):
        """Forward pass through the LLM.

        Args:
            lm_input: (B, T, D) embedded input sequence
            lm_input_len: (B,) lengths

        Returns:
            logits: (B, T, speech_token_size + 200)
        """
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len)
        logits = self.llm_decoder(lm_output)
        return logits

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ):
        """Autoregressive inference to generate speech tokens.

        Returns:
            List of speech token IDs
        """
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len
        text_emb = self.llm.model.embed_tokens(text)

        # Build LM input
        sos_emb = self.speech_embedding(
            torch.tensor([self.sos], device=device)
        ).unsqueeze(0)  # (1, 1, D)
        task_id_emb = self.speech_embedding(
            torch.tensor([self.task_id], device=device)
        ).unsqueeze(0)  # (1, 1, D)

        if prompt_speech_token_len.item() != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text_emb.dtype
            ).to(device)

        lm_input = torch.concat(
            [sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # Calculate min/max length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # Step by step decode
        out_tokens = []
        cache = None
        for i in range(max_len):
            seq_len = (
                lm_input.shape[1]
                if cache is None
                else lm_input.shape[1] + cache[0][0].size(2)
            )
            y_pred, cache = self.llm.forward_one_step(
                lm_input,
                masks=torch.tril(
                    torch.ones(
                        (1, seq_len, seq_len), device=lm_input.device
                    )
                ).to(torch.bool),
                cache=cache,
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # Top-k sampling
            if i < min_len:
                logp[0, self.eos_token] = -float("inf")
            top_ids = torch.topk(logp.squeeze(0), sampling).indices
            top_ids = top_ids[torch.randint(0, sampling, (1,))].item()
            if top_ids in self.stop_token_ids:
                break
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding(
                torch.tensor([[top_ids]], device=device)
            )

        return out_tokens


# ============================================================================
# 2. Flow Matching Component (DiT-based)
# ============================================================================

# --- DiT Sub-modules ---

class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep conditioning embedding: sinusoidal pos emb -> MLP.

    Level1 operators: Linear, Swish
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            Linear(freq_embed_dim, dim, bias=True),
            Swish(),
            Linear(dim, dim, bias=True),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block for text embedding extra modeling.

    Level1 operators: LayerNorm, Linear, GELUAct
    """

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, intermediate_dim, bias=True)
        self.act = GELUAct()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = Linear(intermediate_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class CausalConvPositionEmbedding(nn.Module):
    """Causal convolutional position embedding."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0
) -> torch.Tensor:
    """Precompute rotary position embedding frequencies."""
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    """Get position embedding indices."""
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (
            torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0)
            * scale.unsqueeze(1)
        ).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


def apply_rotary_pos_emb(t, freqs, scale=1.0):
    """Apply rotary position embedding (from x_transformers).

    Uses the x_transformers library's implementation for correctness.
    """
    from x_transformers.x_transformers import apply_rotary_pos_emb as _apply_rope
    return _apply_rope(t, freqs, scale)


class TextEmbedding(nn.Module):
    """Text embedding with optional ConvNeXt-V2 extra modeling.

    Level1 operators: Embedding
    """

    def __init__(self, text_num_embeds: int, text_dim: int, conv_layers: int = 0, conv_mult: int = 2):
        super().__init__()
        self.text_embed = Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long, device=text.device)
            pos_idx = get_pos_embed_indices(
                batch_start, seq_len, max_pos=self.precompute_max_pos
            )
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)

        return text


class InputEmbedding(nn.Module):
    """Noised input audio and context mixing embedding.

    Level1 operators: Linear
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, spk_dim: int = None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = Linear(mel_dim * 2 + text_dim + spk_dim, out_dim, bias=True)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = spks.unsqueeze(1).expand(-1, x.shape[1], -1)
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Norm Zero for DiT blocks.

    Level1 operators: LayerNorm, Linear, Swish
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = Swish()
        self.linear = Linear(dim, dim * 6, bias=True)
        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    """Adaptive Layer Norm Zero for final DiT layer.

    Level1 operators: LayerNorm, Linear, Swish
    """

    def __init__(self, dim: int):
        super().__init__()
        self.silu = Swish()
        self.linear = Linear(dim, dim * 2, bias=True)
        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class DiTFeedForward(nn.Module):
    """Feed-forward block for DiT.

    Level1 operators: Linear, GELUAct, Dropout
    """

    def __init__(self, dim: int, dim_out: int = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.ff = nn.Sequential(
            Linear(dim, inner_dim, bias=True),
            GELUAct(approximate="tanh"),
            Dropout(dropout),
            Linear(inner_dim, dim_out, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DiTAttention(nn.Module):
    """Self-attention for DiT blocks with rotary position embedding.

    Level1 operators: Linear, ScaledDotProductAttention, Dropout
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dim_head = dim_head

        self.to_q = Linear(dim, self.inner_dim, bias=True)
        self.to_k = Linear(dim, self.inner_dim, bias=True)
        self.to_v = Linear(dim, self.inner_dim, bias=True)
        self.to_out_linear = Linear(self.inner_dim, dim, bias=True)
        self.to_out_dropout = Dropout(dropout)

        self.sdpa = ScaledDotProductAttention(mode="sdpa")

    def forward(self, x, mask=None, rope=None):
        batch_size = x.shape[0]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale = xpos_scale if xpos_scale is not None else 1.0
            k_xpos_scale = xpos_scale ** -1.0 if xpos_scale is not None else 1.0
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Reshape to (B, H, T, D)
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # Build attention mask
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.expand(
                    batch_size, self.heads, query.shape[-2], key.shape[-2]
                )
            # Convert bool mask to float mask for SDPA
            attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
            attn_mask_float = attn_mask_float.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_mask_float = None

        # Use level1 ScaledDotProductAttention
        x = self.sdpa(query, key, value, attn_mask=attn_mask_float)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
        x = x.to(query.dtype)

        x = self.to_out_linear(x)
        x = self.to_out_dropout(x)

        if mask is not None:
            if mask.dim() == 2:
                out_mask = mask.unsqueeze(-1)
            else:
                out_mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~out_mask, 0.0)

        return x


class DiTBlock(nn.Module):
    """DiT Transformer Block with AdaLN-Zero modulation.

    Level1 operators: LayerNorm (via AdaLayerNormZero), Linear, GELUAct,
                      ScaledDotProductAttention, Dropout
    """

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = DiTAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = DiTFeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(self, x, t, mask=None, rope=None):
        # Pre-norm & modulation
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # Feed-forward with modulation
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class DiT(nn.Module):
    """Diffusion Transformer backbone for flow matching.

    Level1 operators: Linear, LayerNorm, GELUAct, Swish, Embedding,
                      ScaledDotProductAttention, Dropout
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.0,
        ff_mult: int = 4,
        mel_dim: int = 80,
        mu_dim: int = None,
        long_skip_connection: bool = False,
        spk_dim: int = None,
        out_channels: int = None,
        static_chunk_size: int = 50,
        num_decoding_left_chunks: int = 2,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        # Rotary embedding (from x_transformers)
        from x_transformers.x_transformers import RotaryEmbedding
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = (
            Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        )

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = Linear(dim, mel_dim, bias=True)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        if streaming is True:
            attn_mask = add_optional_chunk_mask(
                x, mask.bool(), False, False, 0, self.static_chunk_size, -1
            ).unsqueeze(dim=1)
        else:
            attn_mask = (
                add_optional_chunk_mask(
                    x, mask.bool(), False, False, 0, 0, -1
                )
                .repeat(1, x.size(1), 1)
                .unsqueeze(dim=1)
            )

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output


class CausalConditionalCFM(nn.Module):
    """Causal Conditional Flow Matching with Euler ODE solver and CFG.

    This is the flow matching decoder that wraps the DiT estimator.
    """

    def __init__(
        self,
        in_channels: int = 80,
        inference_cfg_rate: float = 0.7,
        t_scheduler: str = "cosine",
        estimator: nn.Module = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inference_cfg_rate = inference_cfg_rate
        self.t_scheduler = t_scheduler
        self.estimator = estimator

        # Pre-generate random noise for deterministic causal inference
        # Match original: set_all_random_seed(0) then torch.randn
        import random as _random
        _random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.register_buffer(
            "rand_noise", torch.randn([1, 80, 50 * 300]), persistent=False
        )

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor = None,
        cond: torch.Tensor = None,
        streaming: bool = False,
    ):
        """Forward diffusion (inference).

        Args:
            mu: (B, n_feats, mel_timesteps) encoder output
            mask: (B, 1, mel_timesteps) output mask
            n_timesteps: number of diffusion steps
            temperature: noise scaling temperature
            spks: (B, spk_emb_dim) speaker embedding
            cond: (B, n_feats, mel_timesteps) conditioning

        Returns:
            sample: (B, n_feats, mel_timesteps)
        """
        z = self.rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype) * temperature

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming
        ), None

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """Fixed Euler solver for ODEs with classifier-free guidance."""
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []
        n_feats = x.size(1)
        mu_feats = mu.size(1)
        spk_dim = spks.size(1)

        x_in = torch.zeros([2, n_feats, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, mu_feats, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, spk_dim], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, n_feats, x.size(2)], device=x.device, dtype=spks.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond

            dphi_dt = self.estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(
                dphi_dt, [x.size(0), x.size(0)], dim=0
            )
            dphi_dt = (
                (1.0 + self.inference_cfg_rate) * dphi_dt
                - self.inference_cfg_rate * cfg_dphi_dt
            )
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer for the flow model.

    Two Conv1d layers with residual connection.
    Matches cosyvoice/transformer/upsample_encoder.py::PreLookaheadLayer.
    """

    def __init__(self, in_channels: int = 80, channels: int = 1024, pre_lookahead_len: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: (B, T, C) input tensor
            context: (B, pre_lookahead_len, C) optional lookahead context
        Returns:
            (B, T, C) output tensor
        """
        outputs = inputs.transpose(1, 2).contiguous()
        if context is not None:
            context = context.transpose(1, 2).contiguous()

        # Look ahead
        if context is None or (isinstance(context, torch.Tensor) and context.size(2) == 0):
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        else:
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(
                torch.concat([outputs, context], dim=2),
                (0, self.pre_lookahead_len - context.size(2)),
                mode='constant', value=0.0,
            )

        outputs = F.leaky_relu(self.conv1(outputs))

        # Causal conv2
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # Residual connection
        outputs = outputs + inputs
        return outputs


class CausalMaskedDiffWithDiT(nn.Module):
    """Flow matching model that converts speech tokens to mel spectrograms.

    Level1 operators: Embedding, Linear
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # Level1 operators
        self.input_embedding = Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = Linear(spk_embed_dim, output_size, bias=True)

        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder

    def forward(self, batch: dict, device: torch.device):
        """Training forward pass."""
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        streaming = torch.rand(1).item() < 0.5

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        conds = torch.zeros(feat.shape, device=token.device)
        conds = conds.transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {"loss": loss}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming=False,
        finalize=True,
    ):
        """Inference: convert speech tokens to mel spectrogram."""
        assert token.shape[0] == 1

        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token = torch.concat([prompt_token, token], dim=1)
        token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(
                token[:, : -self.pre_lookahead_len],
                context=token[:, -self.pre_lookahead_len :],
            )
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros(
            [1, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


# ============================================================================
# 3. Vocoder Component (CausalHiFTGenerator)
# ============================================================================

class Snake(nn.Module):
    """Snake activation function: x + (1/a) * sin^2(a*x)."""

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class CausalConv1d(nn.Conv1d):
    """Causal Conv1d with left or right padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        causal_type="left",
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=dilation, groups=groups, bias=bias,
        )
        assert stride == 1
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        assert causal_type in ["left", "right"]
        self.causal_type = causal_type

    def forward(self, x, cache=None):
        if cache is None:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding, device=x.device, dtype=x.dtype)
        if isinstance(cache, torch.Tensor) and cache.numel() == 0:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding, device=x.device, dtype=x.dtype)
        assert cache.size(2) == self.causal_padding
        input_timestep = x.shape[2]
        if self.causal_type == "left":
            x = torch.concat([cache, x], dim=2)
        else:
            x = torch.concat([x, cache], dim=2)
        x = super().forward(x)
        assert x.shape[2] == input_timestep
        return x


class CausalConv1dDownSample(nn.Conv1d):
    """Causal Conv1d with downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1,
                 groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=0, dilation=dilation, groups=groups, bias=bias,
        )
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x, cache=None):
        if cache is None or (isinstance(cache, torch.Tensor) and cache.size(2) == 0):
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super().forward(x)
        return x


class CausalConv1dUpsample(nn.Conv1d):
    """Causal Conv1d with upsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, 1,
            padding=0, dilation=dilation, groups=groups, bias=bias,
        )
        assert dilation == 1
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x, cache=None):
        x = self.upsample(x)
        input_timestep = x.shape[2]
        if cache is None or (isinstance(cache, torch.Tensor) and cache.size(2) == 0):
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super().forward(x)
        assert input_timestep == x.shape[2]
        return x


class VocoderResBlock(nn.Module):
    """Residual block for HiFi-GAN vocoder with Snake activation."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5), causal=False):
        super().__init__()
        self.causal = causal
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm

        for dilation in dilations:
            if causal:
                self.convs1.append(
                    weight_norm(CausalConv1d(channels, channels, kernel_size, 1,
                                             dilation=dilation, causal_type="left"))
                )
                self.convs2.append(
                    weight_norm(CausalConv1d(channels, channels, kernel_size, 1,
                                             dilation=1, causal_type="left"))
                )
            else:
                self.convs1.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                         dilation=dilation, padding=get_padding(kernel_size, dilation)))
                )
                self.convs2.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                         dilation=1, padding=get_padding(kernel_size, 1)))
                )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))]
        )
        self.activations2 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))]
        )

    def forward(self, x):
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class SineGen2(nn.Module):
    """Sine generator for neural source filter (causal version)."""

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003, voiced_threshold=0, causal=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.causal = causal
        if causal:
            self.rand_ini = torch.rand(1, 9)
            self.rand_ini[:, 0] = 0
            self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1

        if not self.training and self.causal:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        else:
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        rad_values = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)

        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="nearest" if self.causal else "linear",
        ).transpose(1, 2)
        sines = torch.sin(phase)
        return sines

    def forward(self, f0):
        fn = torch.multiply(
            f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device)
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if not self.training and self.causal:
            noise = noise_amp * self.sine_waves[:, : sine_waves.shape[1]].to(sine_waves.device)
        else:
            noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """Neural Source Filter source module."""

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0,
                 causal=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen2(
            sampling_rate, upsample_scale, harmonic_num, sine_amp,
            add_noise_std, voiced_threshod, causal=causal,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()
        self.causal = causal
        if causal:
            self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        if not self.training and self.causal:
            noise = self.uv[:, : uv.shape[1]].to(uv.device) * self.sine_amp / 3
        else:
            noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalConvRNNF0Predictor(nn.Module):
    """F0 predictor using causal convolutions."""

    def __init__(self, num_class=1, in_channels=80, cond_channels=512):
        super().__init__()
        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type="right")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x, finalize=True):
        if finalize:
            x = self.condnet[0](x)
        else:
            x = self.condnet[0](
                x[:, :, : -self.condnet[0].causal_padding],
                x[:, :, -self.condnet[0].causal_padding :],
            )
        for i in range(1, len(self.condnet)):
            x = self.condnet[i](x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class CausalHiFTGenerator(nn.Module):
    """
    Causal HiFTNet Generator: Neural Source Filter + ISTFTNet.

    Converts mel spectrograms to raw audio waveforms.

    Level1 operators: Interpolate (for F0 upsampling)
    """

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = None,
        upsample_kernel_sizes: List[int] = None,
        istft_params: Dict[str, int] = None,
        resblock_kernel_sizes: List[int] = None,
        resblock_dilation_sizes: List[List[int]] = None,
        source_resblock_kernel_sizes: List[int] = None,
        source_resblock_dilation_sizes: List[List[int]] = None,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        conv_pre_look_right: int = 4,
        f0_predictor: nn.Module = None,
    ):
        super().__init__()

        if upsample_rates is None:
            upsample_rates = [8, 6, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 12, 4, 4, 4]
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 11, 11, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]

        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates

        # Neural source filter
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=int(np.prod(upsample_rates) * istft_params["hop_len"]),
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            causal=True,
        )

        # F0 upsampling -- level1 Interpolate (nearest mode for 1D)
        self.f0_upsamp = nn.Upsample(
            scale_factor=int(np.prod(upsample_rates) * istft_params["hop_len"])
        )

        # Conv pre
        self.conv_pre = weight_norm(
            CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type="right")
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    CausalConv1dUpsample(
                        base_channels // (2 ** i),
                        base_channels // (2 ** (i + 1)),
                        k, u,
                    )
                )
            )

        # Source downsampling
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    CausalConv1d(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        1, 1, causal_type="left",
                    )
                )
            else:
                self.source_downs.append(
                    CausalConv1dDownSample(
                        istft_params["n_fft"] + 2,
                        base_channels // (2 ** (i + 1)),
                        int(u) * 2, int(u),
                    )
                )
            self.source_resblocks.append(
                VocoderResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True)
            )

        # Main resblocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(VocoderResBlock(ch, k, d, causal=True))

        ch = base_channels // (2 ** len(self.ups))
        self.conv_post = weight_norm(
            CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type="left")
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(
            get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)
        )
        self.conv_pre_look_right = conv_pre_look_right

        # F0 predictor
        self.f0_predictor = f0_predictor if f0_predictor is not None else CausalConvRNNF0Predictor()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse_transform

    def decode(self, x, s=None, finalize=True):
        if s is None:
            s = torch.zeros(1, 1, 0, device=x.device)
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(
                x[:, :, : -self.conv_pre_look_right],
                x[:, :, -self.conv_pre_look_right :],
            )
            s_stft_real = s_stft_real[:, :, : -int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
            s_stft_imag = s_stft_imag[:, :, : -int(np.prod(self.upsample_rates) * self.conv_pre_look_right)]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])

        x = self._istft(magnitude, phase)
        if not finalize:
            x = x[:, : -int(np.prod(self.upsample_rates) * self.istft_params["hop_len"])]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat, finalize=True):
        """Convert mel spectrogram to waveform.

        Args:
            speech_feat: (B, n_mels, T) mel spectrogram
            finalize: whether this is the final chunk

        Returns:
            generated_speech: (B, T_audio) waveform
            s: source signal
        """
        # mel -> f0
        self.f0_predictor.to(torch.float64)
        f0 = self.f0_predictor(speech_feat.to(torch.float64), finalize=finalize).to(speech_feat)
        # f0 -> source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if finalize:
            generated_speech = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            generated_speech = self.decode(
                x=speech_feat[:, :, : -self.f0_predictor.condnet[0].causal_padding],
                s=s, finalize=finalize,
            )
        return generated_speech, s


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    CosyVoice3 Text-to-Speech Model (Fun-CosyVoice3-0.5B-2512).

    Full 3-component TTS pipeline:
    1. CosyVoice3LM: KB-native Qwen2 LLM for speech token generation
    2. CausalMaskedDiffWithDiT: DiT-based flow matching for mel generation
    3. CausalHiFTGenerator: HiFi-GAN vocoder for waveform synthesis

    Level1 operators used:
    - LayerNorm: DiT blocks, AdaLN
    - Linear: projections throughout (LLM decoder, flow embeddings, DiT)
    - GELUAct: DiT feed-forward
    - Swish: timestep embedding MLP
    - Embedding: speech/text embeddings
    - Dropout: attention and feed-forward dropout
    - ScaledDotProductAttention: DiT self-attention
    - Interpolate: F0 upsampling in vocoder

    Supports variant: CosyVoice3-0.5B (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
    """

    VARIANTS = VARIANTS

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__()

        if config is None:
            config = {**DEFAULT_CONFIG, **kwargs}

        # Build LLM component
        self.llm = CosyVoice3LM(
            llm_input_size=config["llm_input_size"],
            llm_output_size=config["llm_output_size"],
            speech_token_size=config["speech_token_size"],
        )

        # Build DiT estimator
        dit = DiT(
            dim=config["dit_dim"],
            depth=config["dit_depth"],
            heads=config["dit_heads"],
            dim_head=config["dit_dim_head"],
            dropout=config["dit_dropout"],
            ff_mult=config["dit_ff_mult"],
            mel_dim=config["dit_mel_dim"],
            mu_dim=config.get("dit_mu_dim", None),
            spk_dim=config["dit_spk_dim"],
            static_chunk_size=config["dit_static_chunk_size"],
            num_decoding_left_chunks=config["dit_num_decoding_left_chunks"],
        )

        # Build CFM decoder
        cfm_decoder = CausalConditionalCFM(
            in_channels=config["flow_output_size"],
            inference_cfg_rate=config["cfm_inference_cfg_rate"],
            t_scheduler=config["cfm_t_scheduler"],
            estimator=dit,
        )

        # Build pre-lookahead layer
        pre_lookahead = PreLookaheadLayer(
            in_channels=config.get("pre_lookahead_in_channels", config["flow_input_size"]),
            channels=config.get("pre_lookahead_channels", 1024),
            pre_lookahead_len=config["flow_pre_lookahead_len"],
        )

        # Build Flow component
        self.flow = CausalMaskedDiffWithDiT(
            input_size=config["flow_input_size"],
            output_size=config["flow_output_size"],
            spk_embed_dim=config["flow_spk_embed_dim"],
            vocab_size=config["flow_vocab_size"],
            token_mel_ratio=config["flow_token_mel_ratio"],
            pre_lookahead_len=config["flow_pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm_decoder,
        )

        # Build Vocoder component
        self.hift = CausalHiFTGenerator(
            in_channels=config["hift_in_channels"],
            base_channels=config["hift_base_channels"],
            nb_harmonics=config["hift_nb_harmonics"],
            sampling_rate=config["hift_sampling_rate"],
            upsample_rates=config["hift_upsample_rates"],
            upsample_kernel_sizes=config["hift_upsample_kernel_sizes"],
            istft_params={
                "n_fft": config["hift_istft_n_fft"],
                "hop_len": config["hift_istft_hop_len"],
            },
            resblock_kernel_sizes=config["hift_resblock_kernel_sizes"],
            resblock_dilation_sizes=config["hift_resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config["hift_source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config["hift_source_resblock_dilation_sizes"],
            conv_pre_look_right=config["hift_conv_pre_look_right"],
        )

    def forward(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        speech_token: torch.Tensor,
        speech_token_len: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_feat_len: torch.Tensor,
        embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass (placeholder - actual training uses component-level losses)."""
        # LLM forward
        lm_input = self.llm.speech_embedding(speech_token)
        logits = self.llm.forward(lm_input, speech_token_len)

        # Flow forward
        flow_batch = {
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "embedding": embedding,
        }
        flow_out = self.flow(flow_batch, text.device)

        return {"logits": logits, "flow_loss": flow_out["loss"]}

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor = None,
        prompt_text_len: torch.Tensor = None,
        prompt_speech_token: torch.Tensor = None,
        prompt_speech_token_len: torch.Tensor = None,
        flow_prompt_speech_token: torch.Tensor = None,
        prompt_speech_feat: torch.Tensor = None,
        embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full TTS inference pipeline: text -> speech tokens -> mel -> waveform.

        Returns:
            waveform: (1, T_audio) generated speech
        """
        device = text.device

        if prompt_text is None:
            prompt_text = torch.zeros(1, 0, dtype=torch.int32, device=device)
        if prompt_text_len is None:
            prompt_text_len = torch.tensor([0], dtype=torch.int32, device=device)
        if prompt_speech_token is None:
            prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32, device=device)
        if prompt_speech_token_len is None:
            prompt_speech_token_len = torch.tensor([0], dtype=torch.int32, device=device)
        if flow_prompt_speech_token is None:
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32, device=device)
        if prompt_speech_feat is None:
            prompt_speech_feat = torch.zeros(1, 0, 80, device=device)
        if embedding is None:
            embedding = torch.zeros(1, 192, device=device)

        # Step 1: LLM generates speech tokens
        speech_tokens = self.llm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=embedding,
        )

        if len(speech_tokens) == 0:
            return torch.zeros(1, 1, device=device)

        speech_token_tensor = torch.tensor(speech_tokens, device=device).unsqueeze(0)

        # Step 2: Flow model converts tokens to mel spectrogram
        tts_mel, _ = self.flow.inference(
            token=speech_token_tensor,
            token_len=torch.tensor([speech_token_tensor.shape[1]], dtype=torch.int32, device=device),
            prompt_token=flow_prompt_speech_token,
            prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=device),
            prompt_feat=prompt_speech_feat,
            prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=device),
            embedding=embedding,
            finalize=True,
        )

        # Step 3: Vocoder converts mel to waveform
        tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=True)

        return tts_speech
