import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable
import threading
from queue import Queue

class Model(nn.Module):
    """
    Asynchronous Communication for Compute-Communication Overlap.
    
    Enables overlapping computation with communication operations
    for improved throughput. Essential for efficient distributed training.
    
    Based on: PyTorch Distributed and NCCL async operations
    """
    def __init__(self, world_size, rank):
        """
        :param world_size: Number of ranks
        :param rank: Current rank
        """
        super(Model, self).__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Simulated communication buffers
        self.send_buffers = {}
        self.recv_buffers = {}
        self.pending_ops = Queue()
    
    def isend(self, tensor, dst, tag=0):
        """
        Non-blocking send.
        
        :param tensor: Tensor to send
        :param dst: Destination rank
        :param tag: Message tag
        :return: Handle for wait()
        """
        handle = {'type': 'send', 'tensor': tensor.clone(), 'dst': dst, 
                  'tag': tag, 'complete': threading.Event()}
        self.pending_ops.put(handle)
        return handle
    
    def irecv(self, tensor, src, tag=0):
        """
        Non-blocking receive.
        
        :param tensor: Buffer to receive into
        :param src: Source rank
        :param tag: Message tag
        :return: Handle for wait()
        """
        handle = {'type': 'recv', 'tensor': tensor, 'src': src,
                  'tag': tag, 'complete': threading.Event()}
        self.pending_ops.put(handle)
        return handle
    
    def wait(self, handle):
        """Wait for async operation to complete."""
        handle['complete'].wait()
        return handle.get('result', None)
    
    def all_reduce_async(self, tensor, op='sum'):
        """
        Non-blocking all-reduce.
        
        :param tensor: Tensor to reduce
        :param op: Reduction operation
        :return: Handle for wait()
        """
        handle = {'type': 'all_reduce', 'tensor': tensor.clone(),
                  'op': op, 'complete': threading.Event(), 'result': None}
        
        # Simulate async all-reduce
        def async_reduce():
            # In real impl: NCCL all-reduce
            result = tensor.clone()  # Simplified
            handle['result'] = result
            handle['complete'].set()
        
        thread = threading.Thread(target=async_reduce)
        thread.start()
        
        return handle
    
    def forward(self, compute_fn, tensors_to_communicate, communication_fn):
        """
        Overlap computation with communication.
        
        :param compute_fn: Function to compute (local work)
        :param tensors_to_communicate: Tensors for communication
        :param communication_fn: Communication function
        :return: Tuple of (compute_result, communication_result)
        """
        # Start async communication
        comm_handles = []
        for tensor in tensors_to_communicate:
            handle = self.all_reduce_async(tensor)
            comm_handles.append(handle)
        
        # Do local computation while communication is in flight
        compute_result = compute_fn()
        
        # Wait for communication
        comm_results = [self.wait(h) for h in comm_handles]
        
        return compute_result, comm_results


# Bucketed All-Reduce with Overlap
class BucketedAllReduce(nn.Module):
    """
    Bucketed Gradient All-Reduce with computation overlap.
    
    Groups gradients into buckets and overlaps all-reduce with
    backward computation.
    """
    def __init__(self, world_size, bucket_size_mb=25):
        super(BucketedAllReduce, self).__init__()
        self.world_size = world_size
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.buckets = []
        self.current_bucket = []
        self.current_bucket_size = 0
    
    def add_gradient(self, grad):
        """
        Add gradient to current bucket.
        
        :param grad: Gradient tensor
        :return: Handle if bucket is flushed, None otherwise
        """
        grad_size = grad.numel() * grad.element_size()
        
        if self.current_bucket_size + grad_size > self.bucket_size_bytes:
            # Flush current bucket
            handle = self._flush_bucket()
            self.current_bucket = [grad]
            self.current_bucket_size = grad_size
            return handle
        
        self.current_bucket.append(grad)
        self.current_bucket_size += grad_size
        return None
    
    def _flush_bucket(self):
        """Flush current bucket with async all-reduce."""
        if not self.current_bucket:
            return None
        
        # Flatten bucket
        flattened = torch.cat([g.view(-1) for g in self.current_bucket])
        
        # Async all-reduce (simulated)
        handle = {'tensors': self.current_bucket.copy(), 
                  'complete': threading.Event()}
        
        return handle
    
    def forward(self, gradients):
        """
        Process all gradients with bucketed all-reduce.
        
        :param gradients: List of gradient tensors
        :return: List of handles
        """
        handles = []
        
        for grad in gradients:
            handle = self.add_gradient(grad)
            if handle is not None:
                handles.append(handle)
        
        # Flush remaining
        final_handle = self._flush_bucket()
        if final_handle is not None:
            handles.append(final_handle)
        
        return handles


# Pipelining Communication and Computation
class ComputeCommPipeline(nn.Module):
    """
    Pipeline stages of compute and communication.
    
    Useful for transformer layers where each layer's output
    can be communicated while next layer computes.
    """
    def __init__(self, world_size, num_stages):
        super(ComputeCommPipeline, self).__init__()
        self.world_size = world_size
        self.num_stages = num_stages
    
    def forward(self, inputs, compute_fns, comm_fns):
        """
        Execute pipelined compute-communication stages.
        
        :param inputs: List of inputs per stage
        :param compute_fns: List of compute functions
        :param comm_fns: List of communication functions
        :return: Final outputs
        """
        # Double buffering for overlap
        buffers = [None, None]
        handles = [None, None]
        
        outputs = []
        
        for stage in range(self.num_stages):
            buf_idx = stage % 2
            prev_buf_idx = (stage - 1) % 2
            
            # Wait for previous communication
            if handles[prev_buf_idx] is not None:
                outputs.append(handles[prev_buf_idx])
            
            # Compute current stage
            if stage < len(inputs):
                buffers[buf_idx] = compute_fns[stage](inputs[stage])
                
                # Start async communication
                if comm_fns[stage] is not None:
                    handles[buf_idx] = comm_fns[stage](buffers[buf_idx])
        
        # Wait for final communication
        if handles[(self.num_stages - 1) % 2] is not None:
            outputs.append(handles[(self.num_stages - 1) % 2])
        
        return outputs


# Test parameters
world_size = 8
rank = 0
tensor_shape = (1024, 1024)

def get_inputs():
    compute_fn = lambda: torch.randn(*tensor_shape) @ torch.randn(*tensor_shape)
    tensors_to_communicate = [torch.randn(*tensor_shape) for _ in range(4)]
    communication_fn = lambda x: x  # Placeholder
    return [compute_fn, tensors_to_communicate, communication_fn]

def get_init_inputs():
    return [world_size, rank]

