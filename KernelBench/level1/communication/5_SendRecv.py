import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task_params import DISTRIBUTIONS, get_supported_distributions
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


PARAMETERS = [
    {"batch_size": 8, "seq_length": 2048, "hidden_size": 4096},
]

SUPPORTED_DISTRIBUTIONS = get_supported_distributions("communication", "5_SendRecv")

def get_inputs(param_idx=0, dist_name=SUPPORTED_DISTRIBUTIONS[0], dtype=torch.float32, device="cuda"):
    assert dist_name in SUPPORTED_DISTRIBUTIONS, f"Distribution {dist_name} not supported"
    assert param_idx < len(PARAMETERS), f"Parameter index {param_idx} out of range"
    p = PARAMETERS[param_idx]
    send_tensor = DISTRIBUTIONS[dist_name]((p["batch_size"], p["seq_length"], p["hidden_size"]), dtype=dtype, device=device)
    return [send_tensor]

def get_init_inputs(param_idx=0, dist_name=None, dtype=None, device=None):
    return []
