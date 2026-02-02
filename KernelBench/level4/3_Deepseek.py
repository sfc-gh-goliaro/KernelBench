"""
DeepSeek-V2 Dense Decoder Model

A decoder-only transformer implementing DeepSeek-V2 architecture with paged KV cache:
- RMSNorm normalization
- Multi-Head Latent Attention (MLA) with low-rank KV compression
- Rotary Position Embeddings (RoPE) with YARN scaling (complex number approach)
- SwiGLU MLP for dense layers
- MoE (Mixture of Experts) with shared experts for layers >= first_k_dense_replace

Implementation aligned with vLLM's paged attention design:
- Each attention layer can own its own KV cache  
- Uses AttentionMetadata for slot_mapping, block_table, etc.
- Supports both prefill and decode phases

Variants:
- DeepSeek-V2-Lite: hidden=2048, heads=16, kv_heads=16, layers=27
- DeepSeek-V2: hidden=5120, heads=128, kv_heads=128, layers=60
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import level1 operators
from ..level1.normalization._4_RMSNorm import Model as RMSNorm
from ..level1.activations._7_Swish import Model as Swish
from ..level1.matmul._10_Linear import Model as Linear
from ..level1.attention._5_MultiHeadLatentAttention import (
    Model as MultiHeadLatentAttention,
    AttentionMetadata,
    create_attention_metadata,
)


# ============================================================================
# Model Variants
# ============================================================================

VARIANTS: Dict[str, str] = {
    "Lite": "deepseek-ai/DeepSeek-V2-Lite",
    "V2": "deepseek-ai/DeepSeek-V2",
}


# ============================================================================
# MoE Components
# ============================================================================

class DeepseekExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMoE(nn.Module):
    """Mixture of Experts layer with shared experts."""
    
    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        n_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        
        # Routed experts as ModuleList (matches HF checkpoint path: mlp.experts.0, mlp.experts.1, ...)
        self.experts = nn.ModuleList([
            DeepseekExpertMLP(hidden_size, moe_intermediate_size)
            for _ in range(n_routed_experts)
        ])
        
        # Router gate
        self.gate = Linear(hidden_size, n_routed_experts, bias=False)
        
        # Shared experts (if any)
        if n_shared_experts is not None and n_shared_experts > 0:
            shared_intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = SwiGLUMLP(hidden_size, shared_intermediate_size)
        else:
            self.shared_experts = None
    
    def route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            router_logits: (batch * seq, num_experts) raw routing scores
            
        Returns:
            topk_idx: (batch * seq, top_k) expert indices
            topk_weight: (batch * seq, top_k) expert weights
        """
        num_tokens = router_logits.shape[0]
        router_probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(router_probs, k=self.num_experts_per_tok, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = router_probs.view(num_tokens, self.n_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(num_tokens, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(num_tokens, -1)
            )
            tmp_scores = router_probs.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        else:
            # Default to greedy
            topk_weight, topk_idx = torch.topk(router_probs, k=self.num_experts_per_tok, dim=-1, sorted=False)
        
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: (batch, seq, hidden_size)
            
        Returns:
            Output tensor (batch, seq, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        residuals = hidden_states
        
        # Compute router logits
        router_logits = F.linear(
            hidden_states.view(-1, hidden_size).float(),
            self.gate.weight.float()
        )
        
        # Route tokens
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        # Process through routed experts
        hidden_flat = hidden_states.view(-1, hidden_size)
        final_hidden_states = torch.zeros_like(hidden_flat)
        
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0].item()
            if expert_idx >= self.n_routed_experts:
                continue
            
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_flat[token_idx]
            
            # Process through expert MLP
            current_hidden_states = self.experts[expert_idx](current_state)
            
            # Apply expert weights (in float32 for precision)
            weighted = current_hidden_states.float() * topk_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))
        
        hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        
        # Add shared expert output
        if self.shared_experts is not None:
            hidden_states = hidden_states + self.shared_experts(residuals)
        
        return hidden_states


# ============================================================================
# Component Modules
# ============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP using level1 operators."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.swish(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DeepseekDecoderLayer(nn.Module):
    """Single DeepSeek decoder layer with paged attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int],
        intermediate_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        n_shared_experts: int,
        num_experts_per_tok: int,
        first_k_dense_replace: int,
        routed_scaling_factor: float,
        topk_method: str,
        n_group: int,
        topk_group: int,
        max_seq_len: int = 163840,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-6,
        block_size: int = 16,
        num_blocks: int = 1024,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.self_attn = MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            block_size=block_size,
            num_blocks=num_blocks,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        
        # Use MoE for layers >= first_k_dense_replace, otherwise dense MLP
        if layer_idx >= first_k_dense_replace:
            self.mlp = DeepseekMoE(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                routed_scaling_factor=routed_scaling_factor,
                topk_method=topk_method,
                n_group=n_group,
                topk_group=topk_group,
            )
        else:
            self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, attn_metadata)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def reset_cache(self):
        """Reset this layer's KV cache."""
        self.self_attn.reset_cache()


# ============================================================================
# Main Model Class
# ============================================================================

class Model(nn.Module):
    """
    DeepSeek-V2 style decoder-only transformer with Paged KV Cache.
    
    Aligned with vLLM's design:
    - Each decoder layer has its own KV cache
    - Uses AttentionMetadata for cache management
    - Supports both prefill and decode phases
    
    Uses level1 operators:
    - RMSNorm (with learnable_weight=True, dim=-1)
    - MultiHeadLatentAttention (with paged KV cache)
    - Linear
    - Swish
    """
    
    def __init__(
        self,
        vocab_size: int = 102400,
        hidden_size: int = 2048,
        num_layers: int = 27,
        num_heads: int = 16,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        kv_lora_rank: int = 512,
        q_lora_rank: Optional[int] = None,
        intermediate_size: int = 10944,
        moe_intermediate_size: int = 1408,
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 6,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        n_group: int = 1,
        topk_group: int = 1,
        max_seq_len: int = 163840,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-6,
        block_size: int = 16,
        num_blocks: int = 1024,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder layers - each with its own KV cache
        self.layers = nn.ModuleList([
            DeepseekDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                intermediate_size=intermediate_size,
                moe_intermediate_size=moe_intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                first_k_dense_replace=first_k_dense_replace,
                routed_scaling_factor=routed_scaling_factor,
                topk_method=topk_method,
                n_group=n_group,
                topk_group=topk_group,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                rms_norm_eps=rms_norm_eps,
                block_size=block_size,
                num_blocks=num_blocks,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        
        # Final norm and LM head
        self.norm = RMSNorm(hidden_size, rms_norm_eps, learnable_weight=True, dim=-1)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache (vLLM-style interface).
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attn_metadata: Attention metadata
            
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embed_tokens(input_ids)

        # Compute position IDs
        if attn_metadata.is_prefill:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = attn_metadata.context_lens.unsqueeze(1) + torch.arange(seq_len, device=device).unsqueeze(0)

        # Process through all layers
        for layer in self.layers:
            x = layer(x, position_ids, attn_metadata)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def reset_cache(self):
        """Reset all layer KV caches."""
        for layer in self.layers:
            layer.reset_cache()

    def _prefill(
        self,
        input_ids: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill (prompt processing) - process all prompt tokens at once."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        context_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )
        
        return self.forward(input_ids, attn_metadata)

    def _decode(
        self,
        input_ids: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Decode (token generation) - process one token at a time."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        assert seq_len == 1, "Decode should process one token at a time"
        
        seq_lens = context_lens + 1
        
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_table=block_table,
            block_size=self.block_size,
            device=device,
        )
        
        return self.forward(input_ids, attn_metadata)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        block_table: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate tokens autoregressively."""
        batch_size, prompt_len = input_ids.shape
        device = input_ids.device
        
        # Reset caches
        self.reset_cache()
        
        # Allocate block table if not provided
        if block_table is None:
            max_seq_len = prompt_len + max(max_new_tokens, 1)
            max_blocks = (max_seq_len + self.block_size - 1) // self.block_size
            block_table = torch.arange(
                max_blocks, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
            for i in range(batch_size):
                block_table[i] = torch.arange(
                    i * max_blocks, 
                    (i + 1) * max_blocks, 
                    device=device
                ) % self.num_blocks
        
        # Prefill
        prefill_logits = self._prefill(input_ids, block_table)
        
        # Handle max_new_tokens=0 case (prefill only)
        if max_new_tokens == 0:
            if return_logits:
                return input_ids, [prefill_logits]
            else:
                return input_ids
        
        all_logits = [prefill_logits] if return_logits else None
        next_token = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        generated = [input_ids, next_token]
        context_lens = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)
        
        # Decode loop
        for _ in range(max_new_tokens - 1):
            decode_logits = self._decode(next_token, block_table, context_lens)
            context_lens += 1
            if return_logits:
                all_logits.append(decode_logits)
            next_token = decode_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
        
        generated_ids = torch.cat(generated, dim=1)
        
        if return_logits:
            return generated_ids, all_logits
        return generated_ids
