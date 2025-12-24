import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Grouped GEMM for MoE Expert Computation.
    
    Performs multiple matrix multiplications with different batch sizes
    in a single kernel launch. Essential for efficient MoE where each
    expert processes a different number of tokens.
    
    Instead of:
        for i, expert in enumerate(experts):
            outputs[i] = expert(inputs[i])
    
    We do:
        outputs = grouped_gemm(inputs, expert_weights, group_sizes)
    
    Reference: Megablocks, Triton grouped GEMM
    """
    def __init__(self, hidden_dim, expert_dim, num_experts):
        """
        :param hidden_dim: Input/output dimension
        :param expert_dim: Expert intermediate dimension
        :param num_experts: Number of experts
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        
        # Expert weights stored contiguously
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim) * 0.02)
    
    def grouped_gemm_scatter(self, x, expert_indices, expert_counts):
        """
        Grouped GEMM with scatter for variable group sizes.
        
        :param x: Permuted tokens (total_active_tokens, hidden_dim)
        :param expert_indices: Expert assignment per token
        :param expert_counts: Number of tokens per expert
        :return: Expert outputs (total_active_tokens, hidden_dim)
        """
        total_tokens = x.shape[0]
        device = x.device
        
        # Compute cumulative counts for indexing
        cumsum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                           expert_counts.cumsum(0)])
        
        # === GROUPED GEMM KERNEL ===
        # First matmul (up-projection)
        intermediate = torch.zeros(total_tokens, self.expert_dim, device=device)
        
        for e in range(self.num_experts):
            start, end = cumsum[e].item(), cumsum[e + 1].item()
            if end > start:
                # x[start:end] @ w1[e] for expert e
                intermediate[start:end] = F.linear(x[start:end], self.w1[e].t())
        
        # Activation
        intermediate = F.silu(intermediate)
        
        # Second matmul (down-projection)
        output = torch.zeros(total_tokens, self.hidden_dim, device=device)
        
        for e in range(self.num_experts):
            start, end = cumsum[e].item(), cumsum[e + 1].item()
            if end > start:
                output[start:end] = F.linear(intermediate[start:end], self.w2[e].t())
        # === END GROUPED GEMM ===
        
        return output
    
    def forward(self, x, expert_indices, expert_counts):
        """
        Forward pass with grouped GEMM.
        
        :param x: Permuted input tokens
        :param expert_indices: Expert indices (for verification)
        :param expert_counts: Token counts per expert
        :return: Expert outputs
        """
        return self.grouped_gemm_scatter(x, expert_indices, expert_counts)


# Megablocks-style batched variant
class GroupedGEMMBatched(nn.Module):
    """
    Batched Grouped GEMM using padding.
    
    Pads groups to uniform size for efficient batched matmul.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts):
        super(GroupedGEMMBatched, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        
        self.w1 = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim) * 0.02)
    
    def forward(self, expert_inputs: List[torch.Tensor]):
        """
        Batched forward with list of expert inputs.
        
        :param expert_inputs: List of (num_tokens_for_expert, hidden_dim) tensors
        :return: List of expert outputs
        """
        device = expert_inputs[0].device if expert_inputs and len(expert_inputs[0]) > 0 else 'cpu'
        
        # Find max tokens for padding
        max_tokens = max(inp.shape[0] for inp in expert_inputs if inp.numel() > 0) if expert_inputs else 1
        
        # Pad and stack
        padded_inputs = []
        masks = []
        
        for e, inp in enumerate(expert_inputs):
            if inp.numel() == 0:
                padded_inputs.append(torch.zeros(max_tokens, self.hidden_dim, device=device))
                masks.append(torch.zeros(max_tokens, dtype=torch.bool, device=device))
            else:
                pad_size = max_tokens - inp.shape[0]
                if pad_size > 0:
                    padded = F.pad(inp, (0, 0, 0, pad_size))
                else:
                    padded = inp
                padded_inputs.append(padded)
                mask = torch.zeros(max_tokens, dtype=torch.bool, device=device)
                mask[:inp.shape[0]] = True
                masks.append(mask)
        
        # Stack: (num_experts, max_tokens, hidden_dim)
        batched_input = torch.stack(padded_inputs)
        
        # Batched matmul
        # (num_experts, max_tokens, hidden_dim) @ (num_experts, hidden_dim, expert_dim)
        intermediate = torch.bmm(batched_input, self.w1)
        intermediate = F.silu(intermediate)
        output = torch.bmm(intermediate, self.w2)
        
        # Unpad
        outputs = []
        for e in range(self.num_experts):
            valid = masks[e].sum().item()
            outputs.append(output[e, :valid])
        
        return outputs


# SwiGLU variant of Grouped GEMM
class GroupedGEMMSwiGLU(nn.Module):
    """
    Grouped GEMM with SwiGLU activation.
    
    Fuses gate/up projection, activation, and down projection.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts):
        super(GroupedGEMMSwiGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        
        # Gate and up packed
        self.w_gate_up = nn.Parameter(torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02)
        self.w_down = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim) * 0.02)
    
    def forward(self, x, expert_indices, expert_counts):
        """Grouped GEMM with SwiGLU."""
        total_tokens = x.shape[0]
        device = x.device
        cumsum = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                           expert_counts.cumsum(0)])
        
        output = torch.zeros(total_tokens, self.hidden_dim, device=device)
        
        for e in range(self.num_experts):
            start, end = cumsum[e].item(), cumsum[e + 1].item()
            if end > start:
                # Fused gate/up
                gate_up = F.linear(x[start:end], self.w_gate_up[e].t())
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = F.silu(gate) * up
                # Down
                output[start:end] = F.linear(hidden, self.w_down[e].t())
        
        return output


# Test parameters
hidden_dim = 4096
expert_dim = 11008
num_experts = 8
total_tokens = 4096

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Simulated permuted tokens with expert assignments
    x = torch.randn(total_tokens, hidden_dim)
    expert_indices = torch.randint(0, num_experts, (total_tokens,))
    expert_counts = torch.bincount(expert_indices, minlength=num_experts)
    # Sort by expert
    sorted_idx = expert_indices.argsort()
    x = x[sorted_idx]
    return [x, expert_indices[sorted_idx], expert_counts]

def get_init_inputs():
    return [hidden_dim, expert_dim, num_experts]

