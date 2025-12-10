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
    Broadcast Collective Communication.
    
    Broadcasts a tensor from one rank (root) to all other ranks.
    Used for distributing model parameters, shared embeddings,
    or synchronized random states.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size, root=0):
        """
        :param world_size: Number of parallel ranks
        :param root: Rank that broadcasts the tensor
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.root = root
    
    def forward(self, tensor):
        """
        Simulate broadcast from root to all ranks.
        
        :param tensor: Tensor from root rank to broadcast
        :return: List of tensors (copies) for each rank
        """
        # Broadcast: all ranks receive copy of root's tensor
        return [tensor.clone() for _ in range(self.world_size)]


# Tree-based Broadcast (latency-optimal)
class TreeBroadcast(nn.Module):
    """
    Tree-based Broadcast for latency optimization.
    
    Uses binary tree topology to reduce latency from O(world_size)
    to O(log(world_size)).
    """
    def __init__(self, world_size, root=0):
        super(TreeBroadcast, self).__init__()
        self.world_size = world_size
        self.root = root
    
    def _build_tree(self):
        """Build binary tree broadcast schedule."""
        # Returns list of (sender, receivers) for each step
        schedule = []
        received = {self.root}
        remaining = set(range(self.world_size)) - received
        
        while remaining:
            step = []
            new_received = set()
            
            for sender in list(received):
                for receiver in list(remaining):
                    if receiver not in new_received:
                        step.append((sender, receiver))
                        new_received.add(receiver)
                        break
            
            received.update(new_received)
            remaining -= new_received
            schedule.append(step)
        
        return schedule
    
    def forward(self, tensor):
        """
        Tree broadcast simulation.
        
        :param tensor: Tensor to broadcast from root
        :return: List of tensors for each rank
        """
        results = [None] * self.world_size
        results[self.root] = tensor.clone()
        
        schedule = self._build_tree()
        
        for step in schedule:
            for sender, receiver in step:
                results[receiver] = results[sender].clone()
        
        return results


# Pipelined Broadcast (bandwidth-optimal for large tensors)
class PipelinedBroadcast(nn.Module):
    """
    Pipelined Broadcast for bandwidth optimization.
    
    Splits tensor into chunks and pipelines them through a ring
    for bandwidth-optimal broadcasting of large tensors.
    """
    def __init__(self, world_size, root=0, num_chunks=4):
        super(PipelinedBroadcast, self).__init__()
        self.world_size = world_size
        self.root = root
        self.num_chunks = num_chunks
    
    def forward(self, tensor):
        """
        Pipelined broadcast.
        
        :param tensor: Tensor to broadcast
        :return: List of tensors for each rank
        """
        # Split into chunks
        chunks = torch.chunk(tensor, self.num_chunks, dim=0)
        
        # Initialize results
        results = [[] for _ in range(self.world_size)]
        results[self.root] = list(chunks)
        
        # Pipeline chunks through ring
        for chunk_idx in range(self.num_chunks):
            for rank in range(self.world_size):
                if rank != self.root:
                    # In ring, receive from previous rank
                    prev_rank = (rank - 1) % self.world_size
                    if prev_rank == self.root or len(results[prev_rank]) > chunk_idx:
                        src = self.root if prev_rank != self.root and len(results[prev_rank]) <= chunk_idx else prev_rank
                        if len(results[src]) > chunk_idx:
                            results[rank].append(results[src][chunk_idx].clone())
        
        # Ensure all ranks have all chunks (simplified simulation)
        for rank in range(self.world_size):
            while len(results[rank]) < self.num_chunks:
                results[rank].append(chunks[len(results[rank])].clone())
        
        # Concatenate chunks
        return [torch.cat(r, dim=0) for r in results]


# Test parameters
world_size = 8
tensor_shape = (2048, 2048)

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Single tensor from root to broadcast
    tensor = torch.randn(*tensor_shape)
    return [tensor]

def get_init_inputs():
    return [world_size, 0]  # world_size, root

