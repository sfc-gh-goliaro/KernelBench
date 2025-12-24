import torch
import torch.nn as nn
import torch.nn.functional as F
import math


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Tree Attention for Speculative Decoding.
    
    Processes multiple draft token sequences (tree structure) in parallel
    using a specialized attention mask that respects tree dependencies.
    
    Based on: "SpecInfer" and "Medusa" papers on speculative decoding
    """
    def __init__(self, dim, num_heads, max_tree_width, max_tree_depth):
        """
        :param dim: Model dimension
        :param num_heads: Number of attention heads
        :param max_tree_width: Maximum width of speculation tree
        :param max_tree_depth: Maximum depth of speculation tree
        """
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_tree_width = max_tree_width
        self.max_tree_depth = max_tree_depth
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def build_tree_mask(self, tree_structure, prefix_len, device):
        """
        Build attention mask for tree-structured speculation.
        
        :param tree_structure: List of parent indices for each tree position
        :param prefix_len: Length of the verified prefix
        :param device: Device to create mask on
        :return: Attention mask (num_tree_nodes, prefix_len + num_tree_nodes)
        """
        num_tree_nodes = len(tree_structure)
        total_len = prefix_len + num_tree_nodes
        
        # Initialize mask (1 = can attend, 0 = cannot attend)
        mask = torch.zeros(num_tree_nodes, total_len, device=device)
        
        # All tree nodes can attend to the prefix
        mask[:, :prefix_len] = 1
        
        # Build tree attention pattern
        for i, parent in enumerate(tree_structure):
            # Can attend to self
            mask[i, prefix_len + i] = 1
            
            # Can attend to ancestors in tree
            current = parent
            while current >= 0:
                mask[i, prefix_len + current] = 1
                if current < len(tree_structure):
                    current = tree_structure[current]
                else:
                    break
        
        return mask
    
    def forward(self, prefix_kv, draft_tokens, tree_structure):
        """
        Forward pass for tree attention.
        
        :param prefix_kv: Tuple of (keys, values) from prefix 
                         Each of shape (batch, num_heads, prefix_len, head_dim)
        :param draft_tokens: Draft token embeddings (batch, num_draft, dim)
        :param tree_structure: Parent indices for tree structure (list)
        :return: Output for draft tokens (batch, num_draft, dim)
        """
        prefix_keys, prefix_values = prefix_kv
        batch_size = draft_tokens.shape[0]
        num_draft = draft_tokens.shape[1]
        prefix_len = prefix_keys.shape[2]
        
        # Project draft tokens to Q, K, V
        q = self.q_proj(draft_tokens)
        k = self.k_proj(draft_tokens)
        v = self.v_proj(draft_tokens)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_draft, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate prefix KV with draft KV
        full_k = torch.cat([prefix_keys, k], dim=2)  # (batch, heads, prefix+draft, head_dim)
        full_v = torch.cat([prefix_values, v], dim=2)
        
        # Build tree attention mask
        tree_mask = self.build_tree_mask(tree_structure, prefix_len, draft_tokens.device)
        tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, draft, prefix+draft)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, full_k.transpose(-2, -1)) * self.scale
        
        # Apply tree mask
        attn_scores = attn_scores.masked_fill(tree_mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.nan_to_num(0)  # Handle all-masked rows
        
        out = torch.matmul(attn_probs, full_v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, num_draft, self.dim)
        
        return self.out_proj(out)


# Test parameters
batch_size = 8
dim = 512
num_heads = 8
prefix_len = 128
max_tree_width = 4
max_tree_depth = 5
num_draft = 16  # Total nodes in speculation tree

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Prefix KV cache
    head_dim = dim // num_heads
    prefix_keys = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    prefix_values = torch.randn(batch_size, num_heads, prefix_len, head_dim)
    
    # Draft tokens
    draft_tokens = torch.randn(batch_size, num_draft, dim)
    
    # Tree structure: parent index for each node (-1 means root)
    # Example: binary tree structure
    tree_structure = [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7][:num_draft]
    
    return [(prefix_keys, prefix_values), draft_tokens, tree_structure]

def get_init_inputs():
    return [dim, num_heads, max_tree_width, max_tree_depth]

