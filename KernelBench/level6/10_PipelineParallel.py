import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Pipeline Parallelism for Model Layers.
    
    Distributes model layers across pipeline stages. Each stage
    processes micro-batches and communicates activations/gradients.
    
    Based on: GPipe and PipeDream
    """
    def __init__(self, num_stages, stage_id, micro_batch_size, num_micro_batches):
        """
        :param num_stages: Total number of pipeline stages
        :param stage_id: Current stage ID (0 = first, num_stages-1 = last)
        :param micro_batch_size: Size of each micro-batch
        :param num_micro_batches: Number of micro-batches per batch
        """
        super(Model, self).__init__()
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = num_micro_batches
        
        # Simple stage model (placeholder)
        self.stage_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
    def send_forward(self, activation):
        """Send activation to next stage."""
        if self.stage_id < self.num_stages - 1:
            return activation.clone()
        return None
    
    def recv_forward(self, activation):
        """Receive activation from previous stage."""
        if self.stage_id > 0:
            return activation.clone()
        return None
    
    def send_backward(self, gradient):
        """Send gradient to previous stage."""
        if self.stage_id > 0:
            return gradient.clone()
        return None
    
    def recv_backward(self, gradient):
        """Receive gradient from next stage."""
        if self.stage_id < self.num_stages - 1:
            return gradient.clone()
        return None
    
    def forward(self, input_activation):
        """
        Process micro-batch through this stage.
        
        :param input_activation: Activation from previous stage
        :return: Activation for next stage
        """
        output = self.stage_model(input_activation)
        return output


# 1F1B Pipeline Schedule (Interleaved)
class PipelineSchedule1F1B(nn.Module):
    """
    1F1B (One Forward One Backward) Pipeline Schedule.
    
    Achieves near-optimal memory efficiency by interleaving
    forward and backward passes.
    """
    def __init__(self, num_stages, stage_id, num_micro_batches):
        super(PipelineSchedule1F1B, self).__init__()
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.num_micro_batches = num_micro_batches
    
    def generate_schedule(self):
        """
        Generate 1F1B schedule for this stage.
        
        :return: List of (micro_batch_id, is_forward) tuples
        """
        schedule = []
        
        # Warmup: forward passes to fill pipeline
        num_warmup = min(self.num_stages - self.stage_id - 1, 
                        self.num_micro_batches)
        
        for mb in range(num_warmup):
            schedule.append((mb, True))  # Forward
        
        # Steady state: 1F1B
        num_steady = self.num_micro_batches - num_warmup
        for mb in range(num_steady):
            schedule.append((num_warmup + mb, True))  # Forward
            schedule.append((mb, False))  # Backward
        
        # Cooldown: remaining backward passes
        for mb in range(num_warmup):
            schedule.append((num_steady + mb, False))
        
        return schedule
    
    def forward(self, micro_batches, stage_outputs=None):
        """
        Execute 1F1B schedule.
        
        :param micro_batches: Dict of micro-batch tensors
        :param stage_outputs: Dict to collect stage outputs (simulation)
        :return: Final outputs for this stage
        """
        schedule = self.generate_schedule()
        
        forward_outputs = {}
        backward_outputs = {}
        
        for mb_id, is_forward in schedule:
            if is_forward:
                if mb_id in micro_batches:
                    forward_outputs[mb_id] = micro_batches[mb_id]
            else:
                if mb_id in forward_outputs:
                    backward_outputs[mb_id] = forward_outputs[mb_id]
        
        return forward_outputs, backward_outputs


# Interleaved Pipeline (Virtual Stages)
class InterleavedPipeline(nn.Module):
    """
    Interleaved Pipeline with Virtual Stages.
    
    Each physical device hosts multiple virtual stages,
    reducing pipeline bubble.
    """
    def __init__(self, num_stages, num_virtual_stages, stage_id):
        super(InterleavedPipeline, self).__init__()
        self.num_stages = num_stages
        self.num_virtual_stages = num_virtual_stages
        self.stage_id = stage_id
        
        # Virtual stages on this physical stage
        self.virtual_stage_ids = [
            stage_id + i * num_stages 
            for i in range(num_virtual_stages)
        ]
        
        # Model chunks for each virtual stage
        self.stage_chunks = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(num_virtual_stages)
        ])
    
    def forward(self, x, virtual_stage_idx):
        """
        Forward through specific virtual stage.
        
        :param x: Input activation
        :param virtual_stage_idx: Which virtual stage (0 to num_virtual_stages-1)
        :return: Output activation
        """
        return self.stage_chunks[virtual_stage_idx](x)


# ZeroBubble Pipeline
class ZeroBubblePipeline(nn.Module):
    """
    Zero Bubble Pipeline Schedule.
    
    Eliminates pipeline bubbles by splitting backward pass
    into B and W phases.
    """
    def __init__(self, num_stages, stage_id, num_micro_batches):
        super(ZeroBubblePipeline, self).__init__()
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.num_micro_batches = num_micro_batches
    
    def generate_zb_schedule(self):
        """
        Generate zero-bubble schedule.
        
        B: Compute input gradients (send to previous stage)
        W: Compute weight gradients (local update)
        """
        schedule = []
        
        # ZB-H1 schedule pattern
        for mb in range(self.num_micro_batches):
            schedule.append((mb, 'F'))  # Forward
        
        for mb in range(self.num_micro_batches):
            schedule.append((mb, 'B'))  # Backward (input grad)
            schedule.append((mb, 'W'))  # Weight grad
        
        return schedule
    
    def forward(self, micro_batches):
        """
        Execute zero-bubble schedule.
        """
        schedule = self.generate_zb_schedule()
        results = {'forward': {}, 'backward': {}, 'weight': {}}
        
        for mb_id, op_type in schedule:
            if op_type == 'F':
                results['forward'][mb_id] = micro_batches.get(mb_id)
            elif op_type == 'B':
                results['backward'][mb_id] = results['forward'].get(mb_id)
            elif op_type == 'W':
                results['weight'][mb_id] = results['forward'].get(mb_id)
        
        return results


# Test parameters
num_stages = 4
stage_id = 0
micro_batch_size = 32
num_micro_batches = 8
hidden_dim = 512

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Micro-batches for pipeline
    micro_batches = {
        i: torch.randn(micro_batch_size, hidden_dim)
        for i in range(num_micro_batches)
    }
    return [micro_batches]

def get_init_inputs():
    return [num_stages, stage_id, micro_batch_size, num_micro_batches]

