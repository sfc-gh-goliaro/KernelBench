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
    Fused MoE All-to-All + Expert GEMM.
    
    Combines the token routing communication with expert computation:
    1. All-to-All dispatch: send tokens to expert ranks
    2. Expert GEMM: compute expert FFN
    3. All-to-All combine: return results to original ranks
    
    Fusion enables overlapping communication with expert computation.
    
    Reference: Megablocks, DeepSpeed-MoE, Tutel
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, world_size, rank,
                 top_k=2, overlap=True):
        """
        :param hidden_dim: Model hidden dimension
        :param expert_dim: Expert intermediate dimension
        :param num_experts: Total number of experts
        :param world_size: Expert parallel world size
        :param rank: Current rank
        :param top_k: Top-k experts per token
        :param overlap: Enable compute-communication overlap
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.top_k = top_k
        self.overlap = overlap
        
        assert num_experts % world_size == 0
        self.experts_per_rank = num_experts // world_size
        
        # Local expert weights
        self.w1 = nn.Parameter(
            torch.randn(self.experts_per_rank, hidden_dim, expert_dim) * 0.02
        )
        self.w2 = nn.Parameter(
            torch.randn(self.experts_per_rank, expert_dim, hidden_dim) * 0.02
        )
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def _route_tokens(self, x):
        """Compute routing and prepare for all-to-all."""
        batch_seq = x.shape[0]
        
        # Gating
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        return indices, weights
    
    def _dispatch_all_to_all(self, x, indices, weights):
        """
        Simulate all-to-all dispatch.
        
        Returns tokens grouped by target rank.
        """
        batch_seq = x.shape[0]
        device = x.device
        
        # Group tokens by destination rank
        dispatched = [[] for _ in range(self.world_size)]
        dispatch_info = [[] for _ in range(self.world_size)]
        
        for token_idx in range(batch_seq):
            for k in range(self.top_k):
                expert_idx = indices[token_idx, k].item()
                dst_rank = expert_idx // self.experts_per_rank
                local_expert = expert_idx % self.experts_per_rank
                
                dispatched[dst_rank].append(x[token_idx])
                dispatch_info[dst_rank].append({
                    'token_idx': token_idx,
                    'local_expert': local_expert,
                    'weight': weights[token_idx, k].item(),
                    'k': k
                })
        
        return dispatched, dispatch_info
    
    def _expert_forward(self, tokens, expert_idx):
        """Compute single expert forward."""
        if len(tokens) == 0:
            return None
        
        x = torch.stack(tokens)
        # SwiGLU-style would have gate and up, but simplified here
        h = F.silu(F.linear(x, self.w1[expert_idx]))
        return F.linear(h, self.w2[expert_idx])
    
    def forward(self, x, all_rank_tokens=None):
        """
        Fused MoE all-to-all + GEMM.
        
        :param x: Input tokens (batch * seq, hidden_dim)
        :param all_rank_tokens: For distributed simulation
        :return: Expert outputs
        """
        batch_seq = x.shape[0]
        device = x.device
        
        # === FUSED KERNEL START ===
        # Step 1: Routing
        indices, weights = self._route_tokens(x)
        
        # Step 2: All-to-All Dispatch
        # In fused kernel, this overlaps with computation
        dispatched, dispatch_info = self._dispatch_all_to_all(x, indices, weights)
        
        # Step 3: Expert Computation (this rank's experts)
        # Process tokens received for each local expert
        expert_outputs = {}
        for expert_idx in range(self.experts_per_rank):
            # Collect tokens for this expert
            expert_tokens = []
            expert_info = []
            
            for token, info in zip(dispatched[self.rank], dispatch_info[self.rank]):
                if info['local_expert'] == expert_idx:
                    expert_tokens.append(token)
                    expert_info.append(info)
            
            if expert_tokens:
                outputs = self._expert_forward(expert_tokens, expert_idx)
                expert_outputs[expert_idx] = (outputs, expert_info)
        
        # Step 4: All-to-All Combine
        # Return results to original ranks
        output = torch.zeros(batch_seq, self.hidden_dim, device=device)
        
        for expert_idx, (outputs, info_list) in expert_outputs.items():
            for i, info in enumerate(info_list):
                token_idx = info['token_idx']
                weight = info['weight']
                output[token_idx] += weight * outputs[i]
        # === FUSED KERNEL END ===
        
        return output


# Overlapped MoE with chunked dispatch
class FusedOverlappedMoE(nn.Module):
    """
    MoE with overlapped dispatch/compute/combine.
    
    Chunks the all-to-all and overlaps with expert computation.
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, world_size, rank,
                 num_chunks=4):
        super(FusedOverlappedMoE, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.num_chunks = num_chunks
        
        self.experts_per_rank = num_experts // world_size
        
        self.w1 = nn.Parameter(
            torch.randn(self.experts_per_rank, hidden_dim, expert_dim) * 0.02
        )
        self.w2 = nn.Parameter(
            torch.randn(self.experts_per_rank, expert_dim, hidden_dim) * 0.02
        )
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x):
        """
        Overlapped MoE forward.
        
        Pipeline:
        - Chunk 0: dispatch while computing routing
        - Chunk 1-N: dispatch chunk i while computing chunk i-1
        - Final: combine while computing last chunk
        """
        batch_seq = x.shape[0]
        chunk_size = batch_seq // self.num_chunks
        device = x.device
        
        output = torch.zeros_like(x)
        
        for chunk_idx in range(self.num_chunks):
            start = chunk_idx * chunk_size
            end = (chunk_idx + 1) * chunk_size if chunk_idx < self.num_chunks - 1 else batch_seq
            
            chunk_x = x[start:end]
            
            # Routing for chunk
            logits = self.gate(chunk_x)
            weights, indices = torch.topk(logits, 2, dim=-1)
            weights = F.softmax(weights, dim=-1)
            
            # Simplified: apply to first expert only
            for i, token in enumerate(chunk_x):
                expert = indices[i, 0].item() % self.experts_per_rank
                h = F.silu(F.linear(token, self.w1[expert]))
                output[start + i] = weights[i, 0] * F.linear(h, self.w2[expert])
        
        return output


# Test parameters
batch_size = 32
seq_len = 512
hidden_dim = 4096
expert_dim = 11008
num_experts = 64
world_size = 8
rank = 0

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size * seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, expert_dim, num_experts, world_size, rank]

