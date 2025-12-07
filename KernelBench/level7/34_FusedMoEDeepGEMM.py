import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class Model(nn.Module):
    """
    Fused MoE with DeepGEMM Optimization.
    
    SGLang's optimized MoE kernel that achieves nearly constant latency
    when incrementing batch size through:
    1. Grouped GEMM with optimal tiling
    2. Persistent kernels for expert computation
    3. Pre-computed expert assignments for better memory access
    4. Load balancing across SMs
    
    Key insight: By sorting tokens by expert and using grouped GEMM,
    we can achieve batch-independent latency up to certain thresholds.
    
    Reference: SGLang DeepGEMM MoE, Megablocks persistent kernels
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2,
                 expert_capacity_factor=1.0, use_persistent=True):
        """
        :param hidden_dim: Input/output hidden dimension  
        :param expert_dim: Expert intermediate dimension
        :param num_experts: Number of experts
        :param top_k: Top-k experts per token
        :param expert_capacity_factor: Capacity factor for load balancing
        :param use_persistent: Use persistent kernel scheduling
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.use_persistent = use_persistent
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Expert weights (SwiGLU architecture)
        # Packed gate+up for single matmul
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
        
        # Pre-allocated buffers for sorted processing
        self.register_buffer('_sorted_indices_cache', None)
    
    def _compute_routing(self, x):
        """
        Compute routing with load balancing.
        
        Returns sorted tokens and expert assignment info.
        """
        num_tokens = x.shape[0]
        device = x.device
        
        # Gating
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Flatten for processing
        flat_weights = weights.view(-1)
        flat_indices = indices.view(-1)
        token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        # Sort by expert for coalesced access
        sorted_order = flat_indices.argsort(stable=True)
        sorted_experts = flat_indices[sorted_order]
        sorted_tokens = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        
        # Compute expert boundaries (for grouped GEMM)
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            expert_counts.cumsum(0)
        ])
        
        return {
            'sorted_tokens': sorted_tokens,
            'sorted_experts': sorted_experts,
            'sorted_weights': sorted_weights,
            'expert_counts': expert_counts,
            'expert_offsets': expert_offsets
        }
    
    def _grouped_expert_gemm(self, x, routing_info):
        """
        DeepGEMM-style grouped expert computation.
        
        Uses persistent kernel approach for constant latency.
        """
        num_tokens = x.shape[0]
        device = x.device
        
        sorted_tokens = routing_info['sorted_tokens']
        sorted_weights = routing_info['sorted_weights']
        expert_counts = routing_info['expert_counts']
        expert_offsets = routing_info['expert_offsets']
        
        # Prepare batched inputs (all tokens sorted by expert)
        sorted_inputs = x[sorted_tokens]
        
        # === DEEP GEMM KERNEL ===
        # In practice, this is a single fused kernel that:
        # 1. Loads expert weights once
        # 2. Processes all tokens for that expert
        # 3. Uses persistent threads to minimize kernel launch overhead
        
        # Accumulator for outputs
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        # Process experts in parallel (simulated sequentially here)
        expert_outputs = []
        
        for e in range(self.num_experts):
            start = expert_offsets[e].item()
            end = expert_offsets[e + 1].item()
            
            if end <= start:
                continue
            
            # Get tokens for this expert
            expert_inputs = sorted_inputs[start:end]
            expert_weights = sorted_weights[start:end]
            expert_token_ids = sorted_tokens[start:end]
            
            # SwiGLU expert forward (fused gate_up + silu*mul + down)
            gate_up = F.linear(expert_inputs, self.expert_gate_up[e].t())
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_out = F.linear(hidden, self.expert_down[e].t())
            
            # Weighted output
            weighted_out = expert_weights.unsqueeze(-1) * expert_out
            
            # Scatter-add to output
            output.index_add_(0, expert_token_ids, weighted_out)
        
        # === END DEEP GEMM ===
        
        return output
    
    def forward(self, x):
        """
        Forward pass with DeepGEMM optimization.
        
        :param x: Input (batch * seq, hidden_dim)
        :return: Output (batch * seq, hidden_dim)
        """
        # Compute routing once
        routing_info = self._compute_routing(x)
        
        # Grouped expert computation
        output = self._grouped_expert_gemm(x, routing_info)
        
        return output


# With FP8 quantization
class FusedMoEDeepGEMMFP8(nn.Module):
    """
    DeepGEMM MoE with FP8 quantization.
    
    Combines DeepGEMM optimization with FP8 inference.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2):
        super(FusedMoEDeepGEMMFP8, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.fp8_max = 448.0
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # FP8 quantized expert weights (stored as float16 for simulation)
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
        
        # FP8 scales per expert
        self.gate_up_scales = nn.Parameter(torch.ones(num_experts))
        self.down_scales = nn.Parameter(torch.ones(num_experts))
    
    def forward(self, x):
        """FP8 DeepGEMM MoE forward."""
        num_tokens = x.shape[0]
        device = x.device
        
        # FP8 quantize input
        x_absmax = x.abs().max()
        x_scale = self.fp8_max / x_absmax.clamp(min=1e-12)
        x_fp8 = (x * x_scale).clamp(-self.fp8_max, self.fp8_max)
        
        # Routing (on original precision for accuracy)
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Sort tokens by expert
        flat_weights = weights.view(-1)
        flat_indices = indices.view(-1)
        token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        sorted_order = flat_indices.argsort(stable=True)
        sorted_experts = flat_indices[sorted_order]
        sorted_tokens = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        
        expert_counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                    expert_counts.cumsum(0)])
        
        # Prepare sorted inputs
        sorted_inputs = x_fp8[sorted_tokens]
        
        # Process experts with FP8
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        for e in range(self.num_experts):
            start, end = expert_offsets[e].item(), expert_offsets[e + 1].item()
            if end <= start:
                continue
            
            expert_inputs = sorted_inputs[start:end] / x_scale
            expert_weights_batch = sorted_weights[start:end]
            expert_token_ids = sorted_tokens[start:end]
            
            # FP8 expert computation
            gate_up = F.linear(expert_inputs, self.expert_gate_up[e].t())
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_out = F.linear(hidden, self.expert_down[e].t())
            
            output.index_add_(0, expert_token_ids,
                             expert_weights_batch.unsqueeze(-1) * expert_out)
        
        return output


# Batch-size independent variant
class FusedMoEConstantLatency(nn.Module):
    """
    MoE with constant latency across batch sizes.
    
    Uses padded expert batches to ensure consistent compute.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2, 
                 max_tokens_per_expert=256):
        super(FusedMoEConstantLatency, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.max_tokens_per_expert = max_tokens_per_expert
        
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
    
    def forward(self, x):
        """Constant latency MoE with padded batches."""
        num_tokens = x.shape[0]
        device = x.device
        
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Pad each expert's batch to max_tokens_per_expert
        padded_inputs = torch.zeros(
            self.num_experts, self.max_tokens_per_expert, self.hidden_dim, 
            device=device
        )
        padded_weights = torch.zeros(
            self.num_experts, self.max_tokens_per_expert,
            device=device
        )
        token_mapping = torch.full(
            (self.num_experts, self.max_tokens_per_expert), -1,
            dtype=torch.long, device=device
        )
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        
        # Assign tokens to experts (with capacity limit)
        for k in range(self.top_k):
            for t in range(num_tokens):
                e = indices[t, k].item()
                pos = expert_counts[e].item()
                if pos < self.max_tokens_per_expert:
                    padded_inputs[e, pos] = x[t]
                    padded_weights[e, pos] = weights[t, k]
                    token_mapping[e, pos] = t
                    expert_counts[e] += 1
        
        # Batched expert computation (constant size per expert)
        expert_outputs = torch.zeros_like(padded_inputs)
        
        for e in range(self.num_experts):
            gate_up = F.linear(padded_inputs[e], self.expert_gate_up[e].t())
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_outputs[e] = F.linear(hidden, self.expert_down[e].t())
        
        # Scatter results back
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        for e in range(self.num_experts):
            for pos in range(self.max_tokens_per_expert):
                t = token_mapping[e, pos].item()
                if t >= 0:
                    output[t] += padded_weights[e, pos] * expert_outputs[e, pos]
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
hidden_dim = 4096
expert_dim = 11008
num_experts = 8
top_k = 2

def get_inputs():
    return [torch.randn(batch_size * seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, expert_dim, num_experts, top_k]

