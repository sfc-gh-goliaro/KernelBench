import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Model(nn.Module):
    """
    Fused MoE with Quantization (FP8/INT8).
    
    MoE layer where quantization is integrated into the dispatch/compute pipeline:
    1. Quantize tokens before dispatch (optional)
    2. Dispatch to experts (All-to-All with quantized data)
    3. Expert computation with quantized weights
    4. Quantize output before combine (optional)
    5. Combine results (All-to-All back)
    
    Reduces communication bandwidth by transmitting quantized activations.
    
    Reference: vLLM FP8 MoE, DeepSpeed FP8, Mixtral FP8 inference
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2,
                 input_quant='fp8_e4m3', weight_quant='fp8_e4m3',
                 output_quant='fp8_e4m3'):
        """
        :param hidden_dim: Input/output hidden dimension
        :param expert_dim: Expert intermediate dimension
        :param num_experts: Number of experts
        :param top_k: Top-k experts per token
        :param input_quant: Input quantization type (None, 'fp8_e4m3', 'int8')
        :param weight_quant: Weight quantization type
        :param output_quant: Output quantization type
        """
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_quant = input_quant
        self.weight_quant = weight_quant
        self.output_quant = output_quant
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Expert weights (stored quantized if specified)
        if weight_quant:
            # Quantized weights
            self.expert_w1 = nn.Parameter(
                torch.randn(num_experts, hidden_dim, expert_dim) * 0.02
            )
            self.expert_w2 = nn.Parameter(
                torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
            )
            # Weight scales per expert
            self.w1_scales = nn.Parameter(torch.ones(num_experts))
            self.w2_scales = nn.Parameter(torch.ones(num_experts))
        else:
            self.expert_w1 = nn.Parameter(
                torch.randn(num_experts, hidden_dim, expert_dim) * 0.02
            )
            self.expert_w2 = nn.Parameter(
                torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
            )
    
    def _get_fp8_max(self, dtype):
        if dtype == 'fp8_e4m3':
            return 448.0
        elif dtype == 'fp8_e5m2':
            return 57344.0
        else:  # int8
            return 127.0
    
    def _quantize(self, x, quant_type):
        """Quantize tensor."""
        if quant_type is None:
            return x, None
        
        max_val = self._get_fp8_max(quant_type)
        absmax = x.abs().max()
        scale = max_val / absmax.clamp(min=1e-12)
        x_quant = (x * scale).clamp(-max_val, max_val)
        
        if 'int8' in quant_type:
            x_quant = x_quant.round().to(torch.int8)
        else:
            x_quant = x_quant.to(torch.float16)
        
        return x_quant, 1.0 / scale
    
    def _dequantize(self, x_quant, scale):
        """Dequantize tensor."""
        if scale is None:
            return x_quant
        return x_quant.float() * scale
    
    def forward(self, x):
        """
        Fused MoE with quantization.
        
        :param x: Input tokens (batch * seq, hidden_dim)
        :return: Tuple of (output, output_scale)
        """
        num_tokens = x.shape[0]
        device = x.device
        
        # === FUSED KERNEL START ===
        # Step 1: Routing
        logits = self.gate(x)
        routing_weights, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Step 2: Quantize input for dispatch (optional)
        if self.input_quant:
            x_quant, input_scale = self._quantize(x, self.input_quant)
        else:
            x_quant, input_scale = x, None
        
        # Step 3: Expert computation with quantized weights
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        for k in range(self.top_k):
            expert_idx = expert_indices[:, k]
            weight = routing_weights[:, k]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue
                
                # Get tokens for this expert
                expert_tokens = x_quant[mask]
                if input_scale is not None:
                    expert_tokens = self._dequantize(expert_tokens, input_scale)
                
                # Get quantized expert weights
                w1 = self.expert_w1[e]
                w2 = self.expert_w2[e]
                
                if self.weight_quant:
                    w1 = w1 * self.w1_scales[e]
                    w2 = w2 * self.w2_scales[e]
                
                # Expert forward with SwiGLU-style activation
                h = F.silu(F.linear(expert_tokens, w1.t()))
                expert_out = F.linear(h, w2.t())
                
                # Weighted accumulation
                output[mask] += weight[mask].unsqueeze(-1) * expert_out
        
        # Step 4: Quantize output (optional)
        if self.output_quant:
            output, output_scale = self._quantize(output, self.output_quant)
        else:
            output_scale = None
        # === FUSED KERNEL END ===
        
        return output, output_scale


# FP8 MoE with SwiGLU
class FusedMoEFP8SwiGLU(nn.Module):
    """
    Fused MoE with FP8 and SwiGLU experts.
    
    Combines:
    - FP8 input/output quantization
    - SwiGLU expert architecture
    - Quantized dispatch communication
    """
    def __init__(self, hidden_dim, expert_dim, num_experts, top_k=2):
        super(FusedMoEFP8SwiGLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.fp8_max = 448.0
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # SwiGLU experts: gate_up packed, down separate
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_dim, 2 * expert_dim) * 0.02
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_dim, hidden_dim) * 0.02
        )
        
        # FP8 scales
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.output_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """FP8 MoE with SwiGLU."""
        num_tokens = x.shape[0]
        device = x.device
        
        # FP8 quantize input
        x_scale = self.fp8_max / x.abs().max().clamp(min=1e-12)
        x_fp8 = (x * x_scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        
        # Routing
        logits = self.gate(x)  # Use original for routing
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Expert computation
        output = torch.zeros(num_tokens, self.hidden_dim, device=device)
        
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = (indices[:, k] == e)
                if not mask.any():
                    continue
                
                # Dequantize for compute
                tokens = x_fp8[mask].float() / x_scale
                
                # SwiGLU expert
                gate_up = F.linear(tokens, self.expert_gate_up[e].t())
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = F.silu(gate) * up
                expert_out = F.linear(hidden, self.expert_down[e].t())
                
                output[mask] += weights[mask, k].unsqueeze(-1) * expert_out
        
        # FP8 quantize output
        out_scale = self.fp8_max / output.abs().max().clamp(min=1e-12)
        output_fp8 = (output * out_scale).clamp(-self.fp8_max, self.fp8_max).to(torch.float16)
        
        return output_fp8, 1.0 / out_scale


# Test parameters
batch_size = 32
seq_len = 512
hidden_dim = 4096
expert_dim = 11008
num_experts = 8

def get_inputs():
    return [torch.randn(batch_size * seq_len, hidden_dim)]

def get_init_inputs():
    return [hidden_dim, expert_dim, num_experts]

