import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Point-to-Point Send/Receive Communication
    
    Used by: Pipeline parallelism, expert parallelism
    
    Simulates point-to-point tensor communication between processes.
    In real distributed settings, uses NCCL send/recv operations.
    
    This implementation simulates the operation for benchmarking
    the memory and compute patterns involved.
    
    Shapes:
        Input: (batch_size, seq_length, hidden_size)
        Output: (batch_size, seq_length, hidden_size)
    """
    
    def __init__(self, hidden_size: int = 4096):
        """
        Initialize Send/Recv operation.
        
        Args:
            hidden_size: Hidden dimension of tensors being communicated
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        
        # Buffer for received data (simulated)
        self.register_buffer('recv_buffer', torch.zeros(1))
    
    def forward(self, send_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulate send/receive operation.
        
        In actual distributed setting:
        - Send tensor to peer process
        - Receive tensor from peer process
        
        Args:
            send_tensor: Tensor to send (batch_size, seq_length, hidden_size)
            
        Returns:
            Received tensor (same shape as input)
        """
        # In practice, this would be:
        # dist.send(send_tensor, dst=peer_rank)
        # dist.recv(recv_tensor, src=peer_rank)
        
        # Simulate the operation by copying (represents memory bandwidth)
        recv_tensor = send_tensor.clone()
        
        return recv_tensor


# ============================================================================
# Benchmark Configuration
# ============================================================================

batch_size = 8
seq_length = 2048
hidden_size = 4096

def get_inputs():
    """Generate input tensors for forward pass benchmarking."""
    send_tensor = torch.randn(batch_size, seq_length, hidden_size, device='cuda')
    return [send_tensor]

def get_init_inputs():
    """Return initialization arguments for the Model class."""
    return [hidden_size]

