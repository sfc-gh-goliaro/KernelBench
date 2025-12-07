import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Top-K Gating mechanism for Mixture of Experts (MoE).
    
    Routes each token to the top-k experts based on gating scores.
    Includes load balancing auxiliary loss computation.
    
    Based on: "Switch Transformers" and "Mixture of Experts with Expert Choice Routing"
    """
    def __init__(self, dim, num_experts, top_k=2, noise_std=1.0, capacity_factor=1.25):
        """
        :param dim: Input dimension
        :param num_experts: Number of experts
        :param top_k: Number of experts each token is routed to
        :param noise_std: Standard deviation of gating noise during training
        :param capacity_factor: Factor to determine expert capacity
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        # Router (gating network)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
    def _compute_aux_loss(self, gates, mask):
        """
        Compute load balancing auxiliary loss.
        
        Encourages uniform distribution of tokens across experts.
        """
        # gates: (batch * seq, num_experts) - probability of routing to each expert
        # mask: (batch * seq, num_experts) - binary mask indicating which experts were selected
        
        # Fraction of tokens routed to each expert
        tokens_per_expert = mask.float().sum(dim=0)
        total_tokens = mask.shape[0]
        expert_fraction = tokens_per_expert / total_tokens  # f_i
        
        # Average gating probability for each expert
        gate_fraction = gates.sum(dim=0) / total_tokens  # P_i
        
        # Auxiliary loss: sum(f_i * P_i) * num_experts
        aux_loss = (expert_fraction * gate_fraction).sum() * self.num_experts
        
        return aux_loss
    
    def forward(self, x, training=True):
        """
        Forward pass for Top-K gating.
        
        :param x: Input tensor of shape (batch_size, seq_len, dim)
        :param training: Whether in training mode (adds noise)
        :return: Tuple of (expert_indices, expert_weights, aux_loss)
            - expert_indices: (batch_size, seq_len, top_k) - indices of selected experts
            - expert_weights: (batch_size, seq_len, top_k) - weights for selected experts
            - aux_loss: Scalar load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, self.dim)  # (batch * seq, dim)
        
        # Compute gating logits
        logits = self.gate(x_flat)  # (batch * seq, num_experts)
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Compute gating probabilities
        gates = F.softmax(logits, dim=-1)  # (batch * seq, num_experts)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Normalize weights of selected experts
        top_k_weights = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Create mask for aux loss computation
        mask = torch.zeros_like(gates)
        mask.scatter_(1, top_k_indices, 1.0)
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(gates, mask)
        
        # Reshape outputs
        expert_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        expert_weights = top_k_weights.view(batch_size, seq_len, self.top_k)
        
        return expert_indices, expert_weights, aux_loss


# Test parameters
batch_size = 32
seq_len = 512
dim = 768
num_experts = 8
top_k = 2

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, num_experts, top_k]

