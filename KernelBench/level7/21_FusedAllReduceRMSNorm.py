import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused AllReduce + RMSNorm.
    
    Combines tensor-parallel all-reduce with RMSNorm in a single operation.
    The all-reduce output feeds directly into RMSNorm without intermediate storage.
    
    This is critical for tensor-parallel inference where:
    1. Row-parallel linear produces partial sums
    2. All-reduce aggregates across ranks
    3. RMSNorm normalizes for next layer
    
    Reference: vLLM, SGLang tensor parallelism
    """
    def __init__(self, hidden_dim, world_size, eps=1e-6):
        """
        :param hidden_dim: Hidden dimension
        :param world_size: Tensor parallel world size
        :param eps: RMSNorm epsilon
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.eps = eps
        
        # RMSNorm parameters
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def _simulate_all_reduce(self, partial_outputs):
        """Simulate all-reduce sum across ranks."""
        return sum(partial_outputs)
    
    def forward(self, x, all_rank_partials=None):
        """
        Fused AllReduce + RMSNorm.
        
        :param x: Local partial result (batch, seq, hidden_dim)
        :param all_rank_partials: List of partials from all ranks (for simulation)
        :return: Normalized output
        """
        # === FUSED KERNEL START ===
        # Step 1: All-Reduce
        if all_rank_partials is not None:
            reduced = self._simulate_all_reduce(all_rank_partials)
        else:
            # In real distributed setting, this would be nccl all-reduce
            reduced = x
        
        # Step 2: RMSNorm (fused with all-reduce completion)
        # In a fused kernel, RMSNorm computation starts as soon as
        # all-reduce data arrives, hiding communication latency
        variance = reduced.pow(2).mean(dim=-1, keepdim=True)
        output = reduced * torch.rsqrt(variance + self.eps) * self.weight
        # === FUSED KERNEL END ===
        
        return output


# With residual add
class FusedAllReduceResidualRMSNorm(nn.Module):
    """
    Fused AllReduce + Residual Add + RMSNorm.
    
    Full pattern after tensor-parallel attention/FFN:
    output = RMSNorm(residual + AllReduce(partial))
    """
    def __init__(self, hidden_dim, world_size, eps=1e-6):
        super(FusedAllReduceResidualRMSNorm, self).__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, partial, residual, all_rank_partials=None):
        """
        Fused AllReduce + Residual + RMSNorm.
        
        :param partial: Local partial result from TP linear
        :param residual: Residual connection from input
        :param all_rank_partials: For simulation
        :return: Normalized output with residual
        """
        # === FUSED KERNEL START ===
        # All-Reduce
        if all_rank_partials is not None:
            reduced = sum(all_rank_partials)
        else:
            reduced = partial
        
        # Residual add
        hidden = reduced + residual
        
        # RMSNorm
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        output = hidden * torch.rsqrt(variance + self.eps) * self.weight
        # === FUSED KERNEL END ===
        
        return output


# Test parameters
batch_size = 32
seq_len = 2048
hidden_dim = 4096
world_size = 8

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_dim)
    return [x]

def get_init_inputs():
    return [hidden_dim, world_size]

