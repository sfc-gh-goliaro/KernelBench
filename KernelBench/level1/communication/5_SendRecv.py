import torch
import torch.nn as nn
import torch.distributed as dist

class Model(nn.Module):
    """
    Point-to-Point Send/Receive Communication (Distributed)

    Used by: Pipeline parallelism, expert parallelism

    Point-to-point tensor communication between processes.
    Uses NCCL send/recv operations for GPU tensors.

    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, hidden_size)
    """

    def __init__(self, peer_rank: int = None, process_group=None):
        """
        Initialize Send/Recv operation.

        Args:
            peer_rank: Rank to exchange data with. If None, exchanges with
                      (rank + 1) % world_size for send and (rank - 1) % world_size for recv
            process_group: The process group to work on. If None, uses default group
        """
        super(Model, self).__init__()
        self.peer_rank = peer_rank
        self.process_group = process_group

    def forward(self, send_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform send/receive operation.

        Sends tensor to peer and receives tensor from peer.
        Uses non-blocking operations to avoid deadlock.

        Args:
            send_tensor: Tensor to send (batch_size, seq_length, hidden_size)

        Returns:
            Received tensor from peer (same shape as input)
        """
        rank = dist.get_rank(self.process_group)
        world_size = dist.get_world_size(self.process_group)

        # Determine peer ranks for send and receive
        if self.peer_rank is not None:
            send_dst = self.peer_rank
            recv_src = self.peer_rank
        else:
            # Default: ring topology - send to next, receive from previous
            send_dst = (rank + 1) % world_size
            recv_src = (rank - 1) % world_size

        # Allocate receive buffer
        recv_tensor = torch.empty_like(send_tensor)

        # Use non-blocking operations to avoid deadlock
        # Even ranks send first, odd ranks receive first
        if rank % 2 == 0:
            send_op = dist.isend(send_tensor, dst=send_dst, group=self.process_group)
            recv_op = dist.irecv(recv_tensor, src=recv_src, group=self.process_group)
        else:
            recv_op = dist.irecv(recv_tensor, src=recv_src, group=self.process_group)
            send_op = dist.isend(send_tensor, dst=send_dst, group=self.process_group)

        # Wait for operations to complete
        send_op.wait()
        recv_op.wait()

        return recv_tensor


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(42 + rank)
    send_tensor = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [send_tensor]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return []
