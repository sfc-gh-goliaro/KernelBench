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
    Momentum Distillation for Online Self-Training.
    
    Maintains a momentum-updated teacher model for generating
    pseudo-labels during training.
    
    Based on: MoCo, DINO, and similar self-distillation approaches
    """
    def __init__(self, dim, hidden_dim, output_dim, momentum=0.999, temperature=0.07):
        """
        :param dim: Input dimension
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Output/embedding dimension
        :param momentum: EMA coefficient for teacher update
        :param temperature: Temperature for softmax
        """
        super(Model, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.momentum = momentum
        self.temperature = temperature
        
        # Student network
        self.student = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Teacher network (momentum-updated copy)
        self.teacher = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize teacher with student weights
        self._init_teacher()
        
        # Disable gradients for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Centering for preventing collapse
        self.register_buffer('center', torch.zeros(1, output_dim))
        
    def _init_teacher(self):
        """Initialize teacher with student parameters."""
        for student_param, teacher_param in zip(
            self.student.parameters(), 
            self.teacher.parameters()
        ):
            teacher_param.data.copy_(student_param.data)
    
    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with momentum."""
        for student_param, teacher_param in zip(
            self.student.parameters(), 
            self.teacher.parameters()
        ):
            teacher_param.data = (
                self.momentum * teacher_param.data + 
                (1 - self.momentum) * student_param.data
            )
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center for centering teacher outputs."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.momentum * self.center + (1 - self.momentum) * batch_center
    
    def forward(self, x, x_aug=None, update_teacher=True):
        """
        Forward pass for momentum distillation.
        
        :param x: Input tensor (batch, dim) - original view
        :param x_aug: Augmented input (batch, dim) - different view
        :param update_teacher: Whether to update teacher
        :return: Tuple of (loss, student_output, teacher_output)
        """
        # Student forward
        student_out = self.student(x)
        
        if x_aug is None:
            x_aug = x
        
        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_out = self.teacher(x_aug)
            
            # Center and sharpen teacher output
            teacher_out = teacher_out - self.center
            teacher_probs = F.softmax(teacher_out / self.temperature, dim=-1)
            
            # Update center
            self.update_center(teacher_out)
        
        # Student output with higher temperature (for matching)
        student_log_probs = F.log_softmax(student_out / (self.temperature * 2), dim=-1)
        
        # Cross-entropy loss (student matches teacher)
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        
        # Update teacher with momentum
        if update_teacher and self.training:
            self.update_teacher()
        
        return loss, student_out, teacher_out
    
    def get_embedding(self, x, use_teacher=False):
        """Get embedding from student or teacher."""
        if use_teacher:
            with torch.no_grad():
                return self.teacher(x)
        return self.student(x)


# Test parameters
batch_size = 64
dim = 768
hidden_dim = 2048
output_dim = 256

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    x = torch.randn(batch_size, dim)
    x_aug = torch.randn(batch_size, dim)  # Augmented view
    return [x, x_aug]

def get_init_inputs():
    return [dim, hidden_dim, output_dim]

