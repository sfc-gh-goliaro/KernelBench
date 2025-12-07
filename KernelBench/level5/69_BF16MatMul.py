import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    BF16 (Brain Floating Point 16) Matrix Multiplication.
    
    Optimized linear layer using BF16 precision for efficient
    training and inference. Used in many modern LLMs.
    
    Based on: NVIDIA Tensor Core optimizations and mixed precision training
    """
    def __init__(self, in_features, out_features, bias=True, use_mixed_precision=True):
        """
        :param in_features: Input dimension
        :param out_features: Output dimension
        :param bias: Whether to use bias
        :param use_mixed_precision: Whether to use mixed precision (BF16 compute, FP32 master)
        """
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_mixed_precision = use_mixed_precision
        
        # Master weights in FP32
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Cached BF16 weight
        self.register_buffer('weight_bf16', None)
    
    def _to_bf16(self, tensor):
        """Convert tensor to BF16."""
        return tensor.to(torch.bfloat16)
    
    def _bf16_matmul(self, x, weight):
        """Perform matmul in BF16."""
        # Convert inputs to BF16
        x_bf16 = self._to_bf16(x)
        w_bf16 = self._to_bf16(weight)
        
        # Matmul in BF16
        out = torch.matmul(x_bf16, w_bf16.t())
        
        return out
    
    def forward(self, x):
        """
        Forward pass with BF16 computation.
        
        :param x: Input tensor (..., in_features)
        :return: Output tensor (..., out_features)
        """
        if self.use_mixed_precision:
            # Compute in BF16
            out = self._bf16_matmul(x, self.weight)
            
            # Convert back to input dtype for accumulation
            out = out.to(x.dtype)
        else:
            # Standard FP32 computation
            out = F.linear(x, self.weight)
        
        # Add bias (in higher precision)
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def prepare_for_inference(self):
        """Convert weights to BF16 for inference."""
        with torch.no_grad():
            self.weight_bf16 = self._to_bf16(self.weight)
    
    def forward_inference(self, x):
        """
        Inference-optimized forward pass.
        
        :param x: Input tensor
        :return: Output tensor
        """
        if self.weight_bf16 is None:
            self.prepare_for_inference()
        
        x_bf16 = self._to_bf16(x)
        out = torch.matmul(x_bf16, self.weight_bf16.t())
        
        if self.bias is not None:
            out = out + self._to_bf16(self.bias)
        
        return out.to(x.dtype)


# Fused BF16 Linear with activation
class BF16LinearGELU(nn.Module):
    """BF16 Linear with fused GELU activation."""
    def __init__(self, in_features, out_features):
        super(BF16LinearGELU, self).__init__()
        self.linear = Model(in_features, out_features, bias=True)
    
    def forward(self, x):
        return F.gelu(self.linear(x))


# Test parameters
batch_size = 32
seq_len = 512
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]

def get_init_inputs():
    return [in_features, out_features]

