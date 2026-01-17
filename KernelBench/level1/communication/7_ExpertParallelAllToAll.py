import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    Expert Parallel AllToAll (Distributed)

    Used by: DeepSeek-V2/V3, Mixtral with expert parallelism

    AllToAll operation for MoE expert parallelism.
    Redistributes tokens from tensor parallel layout to expert parallel layout.
    Tokens are sent to ranks that own their selected experts.

    Shapes:
        Input: (num_local_tokens, hidden_size) with routing info
        Output: (num_tokens_for_local_experts, hidden_size)
    """

    def __init__(self, num_experts: int = 64, process_group=None):
        """
        Initialize Expert Parallel AllToAll.

        Args:
            num_experts: Total number of experts
            process_group: The expert parallel process group
        """
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.process_group = process_group

    def forward(self, tokens: torch.Tensor,
                expert_indices: torch.Tensor) -> tuple:
        """
        Perform expert parallel all-to-all communication.

        1. Partition tokens by destination expert rank
        2. AllToAll exchange tokens
        3. Each rank receives tokens for its local experts

        Args:
            tokens: Token hidden states (num_tokens, hidden_size)
            expert_indices: Selected expert for each token (num_tokens,)

        Returns:
            Tuple of (redistributed_tokens, local_expert_indices, recv_counts)
        """
        world_size = dist.get_world_size(self.process_group)
        rank = dist.get_rank(self.process_group)
        num_tokens, hidden_size = tokens.shape
        device = tokens.device
        experts_per_rank = self.num_experts // world_size

        # Compute which rank owns each expert
        expert_to_rank = expert_indices // experts_per_rank

        # Count tokens going to each rank
        send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        for r in range(world_size):
            send_counts[r] = (expert_to_rank == r).sum()

        # Exchange counts with all ranks to know how many tokens to receive
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.process_group)

        # Sort tokens by destination rank
        sorted_indices = expert_to_rank.argsort(stable=True)
        sorted_tokens = tokens[sorted_indices]
        sorted_expert_indices = expert_indices[sorted_indices]

        # Prepare send buffers (split by destination rank)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # Total tokens to receive
        total_recv = sum(recv_splits)

        # Create receive buffer
        recv_tokens = torch.empty(total_recv, hidden_size, dtype=tokens.dtype, device=device)

        # Perform all-to-all with variable sizes
        dist.all_to_all_single(
            recv_tokens, sorted_tokens,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group
        )

        # Also exchange expert indices
        recv_expert_indices = torch.empty(total_recv, dtype=expert_indices.dtype, device=device)
        dist.all_to_all_single(
            recv_expert_indices, sorted_expert_indices,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group
        )

        # Convert to local expert indices
        local_expert_indices = recv_expert_indices % experts_per_rank

        return recv_tokens, local_expert_indices, recv_counts


# ============================================================================
# Benchmark Configuration
# ============================================================================

num_tokens = 8192
hidden_size = 4096
num_experts = 64

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    tokens = torch.randn(num_tokens, hidden_size, device='cuda')
    expert_indices = torch.randint(0, num_experts, (num_tokens,), device='cuda')
    return [tokens, expert_indices]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [num_experts]
