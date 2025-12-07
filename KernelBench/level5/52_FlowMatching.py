import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Flow Matching for continuous normalizing flows.
    
    Learns to transform noise to data distribution through
    optimal transport paths. Used in StarFlow and Stable Diffusion 3.
    
    Based on: "Flow Matching for Generative Modeling" and Rectified Flow
    """
    def __init__(self, dim, hidden_dim, num_layers=3, sigma_min=1e-4):
        """
        :param dim: Data dimension
        :param hidden_dim: Hidden layer dimension
        :param num_layers: Number of MLP layers
        :param sigma_min: Minimum noise level
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Velocity network (predicts dx/dt)
        layers = []
        layers.append(nn.Linear(dim + hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, dim))
        self.velocity_net = nn.Sequential(*layers)
    
    def get_train_tuple(self, x0, x1):
        """
        Get training tuple for flow matching.
        
        :param x0: Source samples (noise)
        :param x1: Target samples (data)
        :return: (x_t, t, target_velocity)
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=device)
        
        # Interpolate between x0 and x1 (optimal transport path)
        x_t = (1 - t) * x0 + t * x1
        
        # Target velocity is simply (x1 - x0) for linear interpolation
        target_velocity = x1 - x0
        
        return x_t, t, target_velocity
    
    def forward(self, x_t, t):
        """
        Predict velocity field at (x_t, t).
        
        :param x_t: Current position (batch, dim)
        :param t: Time (batch, 1) in [0, 1]
        :return: Predicted velocity (batch, dim)
        """
        # Embed time
        t_emb = self.time_embed(t)
        
        # Concatenate position and time embedding
        x_input = torch.cat([x_t, t_emb], dim=-1)
        
        # Predict velocity
        velocity = self.velocity_net(x_input)
        
        return velocity
    
    def compute_loss(self, x0, x1):
        """
        Compute flow matching loss.
        
        :param x0: Source samples (noise)
        :param x1: Target samples (data)
        :return: MSE loss between predicted and target velocity
        """
        x_t, t, target_velocity = self.get_train_tuple(x0, x1)
        
        # Predict velocity
        pred_velocity = self.forward(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss
    
    def sample(self, x0, num_steps=50):
        """
        Sample by integrating the velocity field.
        
        :param x0: Starting points (noise)
        :param num_steps: Number of integration steps
        :return: Generated samples
        """
        dt = 1.0 / num_steps
        x = x0
        
        for i in range(num_steps):
            t = torch.full((x.shape[0], 1), i / num_steps, device=x.device)
            
            # Euler integration
            velocity = self.forward(x, t)
            x = x + velocity * dt
        
        return x


# Test parameters
batch_size = 64
dim = 784  # e.g., flattened 28x28 image
hidden_dim = 512
num_layers = 4

def get_inputs():
    x_t = torch.randn(batch_size, dim)
    t = torch.rand(batch_size, 1)
    return [x_t, t]

def get_init_inputs():
    return [dim, hidden_dim, num_layers]

