import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Model(nn.Module):
    """
    Point-to-Point (Send/Recv) Communication.
    
    Direct communication between specific pairs of ranks.
    Used for pipeline parallelism, ring attention, and
    custom communication patterns.
    
    Simulates the communication pattern for benchmarking purposes.
    """
    def __init__(self, world_size):
        """
        :param world_size: Number of parallel ranks
        """
        super(Model, self).__init__()
        self.world_size = world_size
    
    def send_recv(self, tensor, src, dst):
        """
        Simulate send from src to dst.
        
        :param tensor: Tensor to send
        :param src: Source rank
        :param dst: Destination rank
        :return: Received tensor at dst
        """
        return tensor.clone()
    
    def forward(self, tensors, communication_pairs):
        """
        Execute multiple send/recv operations.
        
        :param tensors: Dict mapping rank to tensor
        :param communication_pairs: List of (src, dst) pairs
        :return: Dict mapping rank to received tensor
        """
        results = {}
        
        for src, dst in communication_pairs:
            if src in tensors:
                results[dst] = self.send_recv(tensors[src], src, dst)
        
        return results


# Ring Communication Pattern
class RingCommunication(nn.Module):
    """
    Ring Communication Pattern.
    
    Each rank sends to next and receives from previous.
    Used in ring attention for distributed long-context processing.
    """
    def __init__(self, world_size):
        super(RingCommunication, self).__init__()
        self.world_size = world_size
    
    def forward(self, tensors):
        """
        Ring shift: each rank sends to next, receives from previous.
        
        :param tensors: List of tensors, one per rank
        :return: List of shifted tensors
        """
        # Each rank receives from (rank - 1) % world_size
        shifted = [tensors[(i - 1) % self.world_size].clone() 
                   for i in range(self.world_size)]
        return shifted
    
    def bidirectional_exchange(self, tensors_forward, tensors_backward):
        """
        Bidirectional ring exchange.
        
        :param tensors_forward: Tensors to send forward
        :param tensors_backward: Tensors to send backward
        :return: Tuple of (received_forward, received_backward)
        """
        # Forward: receive from previous
        fwd = [tensors_forward[(i - 1) % self.world_size].clone() 
               for i in range(self.world_size)]
        
        # Backward: receive from next
        bwd = [tensors_backward[(i + 1) % self.world_size].clone() 
               for i in range(self.world_size)]
        
        return fwd, bwd


# Pipeline Send/Recv (for pipeline parallelism)
class PipelineSendRecv(nn.Module):
    """
    Pipeline Stage Communication.
    
    Communication between adjacent pipeline stages.
    Supports micro-batch scheduling for pipeline parallelism.
    """
    def __init__(self, num_stages):
        super(PipelineSendRecv, self).__init__()
        self.num_stages = num_stages
    
    def forward_pass(self, activations, stage_id):
        """
        Forward pass: send activations to next stage.
        
        :param activations: Activations from current stage
        :param stage_id: Current pipeline stage
        :return: Activations for next stage (or None if last stage)
        """
        if stage_id < self.num_stages - 1:
            return activations.clone()
        return None
    
    def backward_pass(self, gradients, stage_id):
        """
        Backward pass: send gradients to previous stage.
        
        :param gradients: Gradients from current stage
        :param stage_id: Current pipeline stage
        :return: Gradients for previous stage (or None if first stage)
        """
        if stage_id > 0:
            return gradients.clone()
        return None
    
    def forward(self, micro_batches, schedule):
        """
        Execute pipeline schedule.
        
        :param micro_batches: List of micro-batch tensors
        :param schedule: List of (stage, micro_batch_id, is_forward) tuples
        :return: Dict of results per stage
        """
        results = {stage: [] for stage in range(self.num_stages)}
        stage_buffers = {stage: {} for stage in range(self.num_stages)}
        
        for stage, mb_id, is_forward in schedule:
            if is_forward:
                if stage == 0:
                    # First stage uses input
                    stage_buffers[stage][mb_id] = micro_batches[mb_id]
                
                if mb_id in stage_buffers[stage]:
                    out = self.forward_pass(stage_buffers[stage][mb_id], stage)
                    if out is not None and stage + 1 < self.num_stages:
                        stage_buffers[stage + 1][mb_id] = out
                    if stage == self.num_stages - 1:
                        results[stage].append(out if out is not None else stage_buffers[stage][mb_id])
        
        return results


# Test parameters
world_size = 8
tensor_shape = (256, 1024)

def get_inputs():
    # Tensors for ring communication
    tensors = [torch.randn(*tensor_shape) for _ in range(world_size)]
    return [tensors]

def get_init_inputs():
    return [world_size]

