import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused LAMB - single-kernel with trust ratio."""
    
    def __init__(self, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-6, wd: float = 0.01):
        super(Model, self).__init__()
        self.lr, self.beta1, self.beta2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
    
    def forward(self, param, grad, m, v, step):
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)
        m_hat = m / (1 - self.beta1 ** step)
        v_hat = v / (1 - self.beta2 ** step)
        update = m_hat / (v_hat.sqrt() + self.eps) + self.wd * param
        p_norm, u_norm = param.norm(), update.norm()
        trust = p_norm / u_norm if p_norm > 0 and u_norm > 0 else 1.0
        return param - self.lr * trust * update, m, v

param_shape = (4096, 4096)
def get_inputs():
    return [torch.randn(*param_shape, device='cuda'), torch.randn(*param_shape, device='cuda'),
            torch.zeros(*param_shape, device='cuda'), torch.zeros(*param_shape, device='cuda'), 100]
def get_init_inputs(): return []

