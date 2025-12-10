import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Complete Mixture of Experts (MoE) Layer.
    
    Combines gating, expert computation, and output aggregation
    into a single unified layer.
    
    Based on: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    """
    def __init__(self, dim, hidden_dim, num_experts, top_k=2, 
                 load_balance_weight=0.01, noise_std=1.0):
        """
        :param dim: Input/output dimension
        :param hidden_dim: Expert hidden dimension
        :param num_experts: Number of experts
        :param top_k: Number of experts per token
        :param load_balance_weight: Weight for auxiliary loss
        :param noise_std: Noise for exploration
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std
        
        # Gating network
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            for _ in range(num_experts)
        ])
        
    def _noisy_top_k_gating(self, x):
        """Compute noisy top-k gating."""
        logits = self.gate(x)
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        
        # Compute softmax
        gates = F.softmax(noisy_logits, dim=-1)
        
        # Select top-k
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Normalize selected gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        return gates, top_k_gates, top_k_indices
    
    def _compute_aux_loss(self, gates, top_k_indices):
        """Compute load balancing auxiliary loss."""
        batch_size, seq_len, _ = gates.shape
        num_tokens = batch_size * seq_len
        
        gates_flat = gates.view(-1, self.num_experts)
        indices_flat = top_k_indices.view(-1, self.top_k)
        
        # Compute fraction of tokens routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                expert_counts[e] += (indices_flat[:, k] == e).float().sum()
        
        f_i = expert_counts / (num_tokens * self.top_k)  # Normalize
        
        # Mean routing probability
        P_i = gates_flat.mean(dim=0)
        
        # Aux loss encourages uniform distribution
        aux_loss = (f_i * P_i).sum() * self.num_experts
        
        return aux_loss * self.load_balance_weight
    
    def forward(self, x):
        """
        Forward pass for MoE layer.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :return: Tuple of (output, aux_loss)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute gating
        gates, top_k_gates, top_k_indices = self._noisy_top_k_gating(x)
        
        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(gates, top_k_indices)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Flatten for processing
        x_flat = x.view(-1, self.dim)
        gates_flat = top_k_gates.view(-1, self.top_k)
        indices_flat = top_k_indices.view(-1, self.top_k)
        output_flat = output.view(-1, self.dim)
        
        # Compute expert outputs
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices_flat[:, k] == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    expert_weight = gates_flat[mask, k:k+1]
                    output_flat[mask] += expert_weight * expert_output
        
        return output, aux_loss


# Test parameters
batch_size = 16
seq_len = 256
dim = 512
hidden_dim = 2048
num_experts = 8
top_k = 2

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, hidden_dim, num_experts, top_k]

