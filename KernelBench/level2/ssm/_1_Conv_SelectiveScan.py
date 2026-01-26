import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """Fused Conv1d + Selective Scan (Mamba)."""
    
    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4):
        super(Model, self).__init__()
        self.d_inner, self.d_state = d_inner, d_state
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.register_buffer('A_log', torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
    
    def forward(self, x, delta, B, C):
        B_sz, S, _ = x.shape
        x = F.silu(self.conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2))
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_x = (delta.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)
        h = torch.zeros(B_sz, self.d_inner, self.d_state, device=x.device)
        outs = []
        for t in range(S):
            h = dA[:, t] * h + dB_x[:, t]
            outs.append((h * C[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(outs, 1) + x * self.D

batch_size, seq_len, d_inner, d_state = 8, 2048, 4096, 16
def get_inputs():
    return [torch.randn(batch_size, seq_len, d_inner, device='cuda'),
            F.softplus(torch.randn(batch_size, seq_len, d_inner, device='cuda')),
            torch.randn(batch_size, seq_len, d_state, device='cuda'),
            torch.randn(batch_size, seq_len, d_state, device='cuda')]
def get_init_inputs(): return [d_inner, d_state]

