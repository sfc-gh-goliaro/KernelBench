"""
EAGLE-3 Speculative Decoding Model

Implements the EAGLE-3 speculative decoding draft head architecture:
- Lightweight draft head (single decoder layer) for fast token prediction
- Dynamic tree construction for parallel verification
- Feature-level speculation with fused hidden states from target model
- Reduced draft vocabulary with token mapping (d2t / t2d)

Reference: https://github.com/SafeAILab/EAGLE

Supported model:
- meta-llama/Llama-3.1-8B-Instruct with draft model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

Architecture summary (EAGLE-3 inference draft head):
1. fc: Linear(3 * target_hidden_size -> hidden_size) to fuse 3 early hidden states from target LLM
2. LlamaDecoderLayeremb: single decoder layer where:
   - input_emb and hidden_states are separately RMSNorm'd
   - concatenated to 2*hidden_size for QKV projections
   - standard GQA attention with RoPE, KV cache
   - SiLU-gated MLP
3. RMSNorm + lm_head (hidden_size -> draft_vocab_size) for draft logits
4. Dynamic tree building via topK_genrate for speculative candidates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-8B-Instruct": {
        "base_model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "eagle_model_name": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "hidden_size": 4096,
        "target_hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "draft_vocab_size": 32000,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
        "hidden_act": "silu",
        "top_k": 10,
        "total_tokens": 60,
        "depth": 7,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# ============================================================================
# Rotary Embedding
# ============================================================================

class EAGLE3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# ============================================================================
# RMSNorm
# ============================================================================

class EAGLE3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# Attention (QKV takes 2*hidden_size input)
# ============================================================================

class EAGLE3Attention(nn.Module):
    """EAGLE-3 attention where Q/K/V projections take 2*hidden_size input
    (concatenation of normalized input_emb and hidden_states)."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = EAGLE3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ============================================================================
# MLP
# ============================================================================

class EAGLE3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# EAGLE-3 Decoder Layer (LlamaDecoderLayeremb from reference)
# ============================================================================

class EAGLE3DecoderLayeremb(nn.Module):
    """Single decoder layer that separately normalizes input_emb and hidden_states,
    concatenates them to 2*hidden_size, then runs attention + MLP."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 intermediate_size, rms_norm_eps=1e-6,
                 max_position_embeddings=2048, rope_theta=10000.0,
                 hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = EAGLE3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = EAGLE3MLP(hidden_size, intermediate_size, hidden_act)
        self.hidden_norm = EAGLE3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm = EAGLE3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = EAGLE3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ============================================================================
# Main EAGLE-3 Draft Model
# ============================================================================

class Model(nn.Module):
    """
    EAGLE-3 speculative decoding draft head model.
    
    This implements the inference-time draft model from the EAGLE-3 paper.
    It is designed to be loaded from pretrained weights from 
    yuhuili/EAGLE3-LLaMA3.1-Instruct-8B.
    
    Architecture:
    - fc: Linear(target_hidden_size * 3 -> hidden_size) to fuse early target hidden states
    - midlayer: Single EAGLE3DecoderLayeremb (takes input_emb + hidden_states)
    - norm: RMSNorm
    - lm_head: Linear(hidden_size -> draft_vocab_size)
    - embed_tokens: Embedding (shared from target model)
    - d2t / t2d: Draft-to-target and target-to-draft vocabulary mappings
    
    Forward pass (for a single step):
    1. Fuse hidden states: hidden = fc(cat(h0, h1, h2))  where h0,h1,h2 are from target layers 0,1,2
    2. Embed input tokens: emb = embed_tokens(input_ids)
    3. Decoder layer: out = midlayer(emb, hidden, attn_mask, position_ids, kv_cache)
    4. Draft logits: logits = lm_head(norm(out))
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        hidden_size: int = 4096,
        target_hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14336,
        vocab_size: int = 128256,
        draft_vocab_size: int = 32000,
        num_hidden_layers: int = 1,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_act: str = "silu",
        top_k: int = 10,
        total_tokens: int = 60,
        depth: int = 7,
        threshold: float = 1.0,
        pad_token_id: Optional[int] = 0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_hidden_size = target_hidden_size
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold) if threshold > 0 else 0.0
        self.padding_idx = pad_token_id

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        self.fc = nn.Linear(target_hidden_size * 3, hidden_size, bias=False)

        self.midlayer = EAGLE3DecoderLayeremb(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            hidden_act=hidden_act,
        )

        self.norm = EAGLE3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t = torch.zeros(draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros(vocab_size, dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        """Initialize tree mask and position_ids for tree attention."""
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(
            self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
        )
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def reset_kv(self):
        self.stable_kv = None

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the EAGLE-3 draft model.

        Args:
            hidden_states: Fused hidden states from target model, shape (batch, seq, target_hidden_size*3)
                           OR already reduced shape (batch, seq, hidden_size) if already passed through fc.
            input_ids: Token ids for embedding lookup, shape (batch, seq)
            attention_mask: Attention mask, shape (batch, seq_with_past)
            position_ids: Position ids, shape (batch, seq)
            past_key_values: KV cache from previous steps
            use_cache: Whether to return KV cache

        Returns:
            hidden_states if use_cache is False, else (hidden_states, next_decoder_cache)
        """
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        next_decoder_cache = () if use_cache else None

        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        """
        Generate tree of draft token candidates using top-K sampling at each depth.

        This is the core speculative decoding tree construction method. It builds
        a dynamic tree of candidate tokens by:
        1. Running the draft model for the initial step to get top-K candidates
        2. Iteratively expanding the tree to the configured depth
        3. Pruning to keep only the top-scoring candidates within budget

        Args:
            hidden_states: Hidden states from the target model (batch, seq, hidden_dim*3 or hidden_dim)
            input_ids: Full input sequence including newly sampled token
            head: Target model's lm_head (not used in EAGLE3 - uses own lm_head)
            logits_processor: Optional logits processor for sampling

        Returns:
            Tuple of (draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
            - draft_tokens: (1, total_tokens+1) candidate token ids
            - retrieve_indices: (num_leaves, max_depth) indices for retrieving candidates
            - tree_mask: (1, 1, total_tokens+1, total_tokens+1) attention mask for tree
            - tree_position_ids: (total_tokens+1,) position ids for each candidate
        """
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values = self(
                hidden_states, input_ids=input_ids, use_cache=True
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.vocab_size == self.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.vocab_size == self.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth_val = position_ids_list[i]
                for j in reversed(range(depth_val + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @classmethod
    def from_pretrained(cls, variant: str = "Llama-3.1-8B-Instruct", **kwargs):
        """Create model from a predefined variant configuration."""
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
        config = VARIANTS[variant].copy()
        config.update(kwargs)
        eagle_model_name = config.pop("eagle_model_name", None)
        base_model_name = config.pop("base_model_name", None)
        model = cls(**config)
        return model

    @classmethod
    def from_eagle_checkpoint(
        cls,
        eagle_model_path: str = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        variant: str = "Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Load the EAGLE-3 draft model from the official checkpoint.

        This loads the draft model weights and the embedding layer from the base model.

        Args:
            eagle_model_path: HuggingFace path to EAGLE-3 draft model
            base_model_path: HuggingFace path to the base (target) LLM
            variant: Configuration variant name
            device: Device to load the model onto
            dtype: Data type for model weights

        Returns:
            Initialized Model with pretrained weights
        """
        import json
        import os
        from huggingface_hub import hf_hub_download

        config = VARIANTS[variant].copy()
        config.pop("eagle_model_name", None)
        config.pop("base_model_name", None)
        model = cls(**config)

        configpath = os.path.join(eagle_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(eagle_model_path, "config.json")

        try:
            load_model_path = os.path.join(eagle_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(eagle_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path, map_location="cpu")
        except Exception:
            from safetensors.torch import load_file
            load_model_path = os.path.join(eagle_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(eagle_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)

        # Load embedding weights from base model
        from safetensors import safe_open
        try:
            index_json_path = os.path.join(base_model_path, "model.safetensors.index.json")
            if not os.path.exists(index_json_path):
                index_json_path = hf_hub_download(base_model_path, "model.safetensors.index.json")
            with open(index_json_path, "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            local_emb_path = os.path.join(base_model_path, emb_path)
            if not os.path.exists(local_emb_path):
                local_emb_path = hf_hub_download(base_model_path, emb_path)
            with safe_open(local_emb_path, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice("model.embed_tokens.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                emb_tensor = tensor_slice[:, :hidden_dim].float()
        except Exception:
            index_json_path = os.path.join(base_model_path, "pytorch_model.bin.index.json")
            if not os.path.exists(index_json_path):
                index_json_path = hf_hub_download(base_model_path, "pytorch_model.bin.index.json")
            with open(index_json_path, "r") as f:
                index_json = json.loads(f.read())
                emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
            local_emb_path = os.path.join(base_model_path, emb_path)
            if not os.path.exists(local_emb_path):
                local_emb_path = hf_hub_download(base_model_path, emb_path)
            weights = torch.load(local_emb_path)
            emb_tensor = weights["model.embed_tokens.weight"].float()

        model.embed_tokens.weight.data = emb_tensor

        if config.get("vocab_size", 128256) == config.get("draft_vocab_size", 32000):
            if hasattr(model, "d2t"):
                del model.d2t
            if hasattr(model, "t2d"):
                del model.t2d

        load_result = model.load_state_dict(ea_layer_state_dict, strict=False)

        model = model.to(dtype).to(device)
        model.eval()
        model.init_tree()

        return model
