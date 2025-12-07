import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Fused Bias + Activation Functions.
    
    Combines bias addition with various activation functions in a single
    kernel pass. This is a common pattern after linear layers.
    
    Supported activations:
    - GELU (Gaussian Error Linear Unit)
    - SiLU/Swish
    - ReLU, LeakyReLU
    - Quick GELU (CLIP-style)
    - Squared ReLU
    
    Reference: cuBLAS epilogue fusion, Apex fused activations
    """
    def __init__(self, dim, activation='gelu', use_bias=True):
        """
        :param dim: Feature dimension
        :param activation: Activation type
        :param use_bias: Whether to include bias
        """
        super(Model, self).__init__()
        self.dim = dim
        self.activation = activation
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('bias', None)
    
    def _apply_activation(self, x):
        """Apply activation function."""
        if self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'gelu_approx' or self.activation == 'gelu_tanh':
            # Approximate GELU using tanh
            return 0.5 * x * (1 + torch.tanh(
                0.7978845608028654 * (x + 0.044715 * x ** 3)
            ))
        elif self.activation == 'quick_gelu':
            # Quick GELU (used in CLIP)
            return x * torch.sigmoid(1.702 * x)
        elif self.activation == 'silu' or self.activation == 'swish':
            return F.silu(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.01)
        elif self.activation == 'squared_relu':
            return F.relu(x) ** 2
        elif self.activation == 'mish':
            return x * torch.tanh(F.softplus(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x):
        """
        Fused bias + activation forward.
        
        :param x: Input tensor (*, dim)
        :return: Activated output (*, dim)
        """
        # === FUSED KERNEL START ===
        if self.bias is not None:
            x = x + self.bias
        x = self._apply_activation(x)
        # === FUSED KERNEL END ===
        
        return x


# Linear + Bias + Activation variant
class FusedLinearBiasActivation(nn.Module):
    """
    Fused Linear + Bias + Activation.
    
    Full fusion of:
    1. Matrix multiplication
    2. Bias addition
    3. Activation function
    
    This is the most common pattern in transformers.
    """
    def __init__(self, in_features, out_features, activation='gelu', bias=True):
        super(FusedLinearBiasActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Fused linear + bias + activation.
        
        In CUDA, this would be a single kernel using cuBLAS epilogue.
        """
        # === FUSED KERNEL START ===
        # Matmul
        output = F.linear(x, self.weight)
        
        # Bias
        if self.bias is not None:
            output = output + self.bias
        
        # Activation
        if self.activation == 'gelu':
            output = F.gelu(output)
        elif self.activation == 'silu':
            output = F.silu(output)
        elif self.activation == 'relu':
            output = F.relu(output)
        # === FUSED KERNEL END ===
        
        return output


# SwiGLU-style fused bias activation
class FusedGatedBiasActivation(nn.Module):
    """
    Fused Gated Bias + Activation (SwiGLU/GeGLU/ReGLU).
    
    For gated activations:
        output = activation(gate + bias_gate) * (value + bias_value)
    """
    def __init__(self, dim, gate_activation='silu', use_bias=True):
        super(FusedGatedBiasActivation, self).__init__()
        self.dim = dim
        self.gate_activation = gate_activation
        
        if use_bias:
            self.gate_bias = nn.Parameter(torch.zeros(dim))
            self.value_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('gate_bias', None)
            self.register_parameter('value_bias', None)
    
    def forward(self, gate, value):
        """
        Fused gated activation.
        
        :param gate: Gate tensor (*, dim)
        :param value: Value tensor (*, dim)
        :return: Gated output (*, dim)
        """
        # === FUSED KERNEL START ===
        # Add biases
        if self.gate_bias is not None:
            gate = gate + self.gate_bias
        if self.value_bias is not None:
            value = value + self.value_bias
        
        # Apply gate activation
        if self.gate_activation == 'silu':
            activated_gate = F.silu(gate)
        elif self.gate_activation == 'gelu':
            activated_gate = F.gelu(gate)
        elif self.gate_activation == 'relu':
            activated_gate = F.relu(gate)
        else:
            activated_gate = gate
        
        # Multiply
        output = activated_gate * value
        # === FUSED KERNEL END ===
        
        return output


# Fused Bias + Activation + Dropout (for training)
class FusedBiasActivationDropout(nn.Module):
    """
    Fused Bias + Activation + Dropout.
    
    Combines bias, activation, and dropout for training efficiency.
    """
    def __init__(self, dim, activation='gelu', dropout_prob=0.1):
        super(FusedBiasActivationDropout, self).__init__()
        self.dim = dim
        self.activation = activation
        self.dropout_prob = dropout_prob
        
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x, training=True):
        """
        Fused bias + activation + dropout.
        """
        # === FUSED KERNEL START ===
        # Bias
        x = x + self.bias
        
        # Activation
        if self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'silu':
            x = F.silu(x)
        
        # Dropout
        if training and self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=True)
        # === FUSED KERNEL END ===
        
        return x


# Test parameters
batch_size = 32
seq_len = 2048
dim = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, dim)]

def get_init_inputs():
    return [dim, 'gelu']

