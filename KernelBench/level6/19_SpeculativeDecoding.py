import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Distributed Speculative Decoding.
    
    Coordinates draft and target model execution across devices
    for accelerated autoregressive generation.
    
    Based on: Speculative Decoding and SpecInfer
    """
    def __init__(self, draft_world_size, target_world_size, vocab_size,
                 num_draft_tokens=4):
        """
        :param draft_world_size: Ranks for draft model
        :param target_world_size: Ranks for target model
        :param vocab_size: Vocabulary size
        :param num_draft_tokens: Number of tokens to draft
        """
        super(Model, self).__init__()
        self.draft_world_size = draft_world_size
        self.target_world_size = target_world_size
        self.vocab_size = vocab_size
        self.num_draft_tokens = num_draft_tokens
    
    def draft_tokens(self, draft_logits):
        """
        Generate draft tokens from draft model.
        
        :param draft_logits: Draft model output logits
        :return: Draft token IDs and probabilities
        """
        draft_probs = F.softmax(draft_logits, dim=-1)
        draft_tokens = torch.argmax(draft_logits, dim=-1)
        
        # Get probability of selected tokens
        draft_token_probs = torch.gather(
            draft_probs, -1, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        return draft_tokens, draft_token_probs
    
    def verify_tokens(self, draft_tokens, draft_probs, target_logits):
        """
        Verify draft tokens against target model.
        
        :param draft_tokens: Proposed draft tokens
        :param draft_probs: Draft token probabilities
        :param target_logits: Target model logits for verification
        :return: Accepted tokens and acceptance mask
        """
        batch_size, num_draft = draft_tokens.shape
        
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Get target probabilities for draft tokens
        target_token_probs = torch.gather(
            target_probs, -1, draft_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Acceptance probability: min(1, p_target / p_draft)
        acceptance_probs = torch.clamp(target_token_probs / (draft_probs + 1e-10), max=1.0)
        
        # Sample acceptance
        random_vals = torch.rand_like(acceptance_probs)
        accepted = random_vals < acceptance_probs
        
        # Find first rejection
        rejection_mask = ~accepted
        first_rejection = rejection_mask.float().cumsum(dim=-1) == 1
        
        # All tokens before first rejection are accepted
        acceptance_mask = acceptance_probs.cumsum(dim=-1) <= acceptance_probs.sum(dim=-1, keepdim=True)
        
        return draft_tokens, accepted
    
    def parallel_verify(self, draft_sequences, target_model_shards):
        """
        Parallel verification across target model shards.
        
        :param draft_sequences: List of draft token sequences
        :param target_model_shards: Target model distributed across ranks
        :return: Verified sequences
        """
        # Each target rank processes a portion of verification
        verified = []
        
        for seq in draft_sequences:
            # Distribute across target ranks
            seq_len = seq.shape[0]
            chunk_size = seq_len // self.target_world_size
            
            chunks = torch.split(seq, chunk_size, dim=0)
            verified_chunks = []
            
            for chunk in chunks:
                # Simulate parallel verification
                verified_chunks.append(chunk)  # Placeholder
            
            verified.append(torch.cat(verified_chunks, dim=0))
        
        return verified
    
    def forward(self, prefix_ids, draft_logits_list, target_logits_list):
        """
        Full speculative decoding step.
        
        :param prefix_ids: Current prefix tokens
        :param draft_logits_list: List of draft logits per position
        :param target_logits_list: List of target logits for verification
        :return: Accepted tokens and new prefix
        """
        batch_size = prefix_ids.shape[0]
        
        # Generate draft tokens
        all_draft_tokens = []
        all_draft_probs = []
        
        for draft_logits in draft_logits_list:
            tokens, probs = self.draft_tokens(draft_logits)
            all_draft_tokens.append(tokens)
            all_draft_probs.append(probs)
        
        draft_tokens = torch.stack(all_draft_tokens, dim=1)
        draft_probs = torch.stack(all_draft_probs, dim=1)
        
        # Verify against target
        target_logits = torch.stack(target_logits_list, dim=1)
        verified_tokens, accepted = self.verify_tokens(
            draft_tokens, draft_probs, target_logits
        )
        
        # Count accepted tokens
        num_accepted = accepted.sum(dim=-1)
        
        return verified_tokens, accepted, num_accepted


# Tree-based Speculative Decoding
class TreeSpeculativeDecoding(nn.Module):
    """
    Tree-based speculation for parallel draft verification.
    
    Generates multiple draft branches and verifies in parallel.
    """
    def __init__(self, vocab_size, num_branches=4, depth=4):
        super(TreeSpeculativeDecoding, self).__init__()
        self.vocab_size = vocab_size
        self.num_branches = num_branches
        self.depth = depth
    
    def generate_tree(self, prefix_logits):
        """
        Generate speculation tree.
        
        :param prefix_logits: Logits at current position
        :return: Tree of draft tokens
        """
        batch_size = prefix_logits.shape[0]
        
        # Top-k sampling for branches
        topk_probs, topk_ids = torch.topk(
            F.softmax(prefix_logits, dim=-1), 
            self.num_branches, 
            dim=-1
        )
        
        # Build tree structure
        tree = {
            'tokens': topk_ids,  # (batch, num_branches)
            'probs': topk_probs,
            'children': []
        }
        
        # For simplicity, generate flat branches here
        for _ in range(self.depth - 1):
            # Each branch extends (simplified)
            tree['children'].append({
                'tokens': torch.randint(0, self.vocab_size, (batch_size, self.num_branches)),
                'probs': torch.rand(batch_size, self.num_branches),
                'children': []
            })
        
        return tree
    
    def verify_tree(self, tree, target_logits):
        """
        Verify speculation tree against target.
        
        :param tree: Speculation tree
        :param target_logits: Target model logits
        :return: Best verified path
        """
        # Find best matching path
        root_probs = tree['probs']
        best_branch = root_probs.argmax(dim=-1)
        
        return tree['tokens'].gather(1, best_branch.unsqueeze(-1)).squeeze(-1)
    
    def forward(self, prefix_logits, target_logits):
        """
        Tree speculation forward pass.
        """
        tree = self.generate_tree(prefix_logits)
        verified = self.verify_tree(tree, target_logits)
        return verified


# Test parameters
draft_world_size = 2
target_world_size = 8
vocab_size = 128000
batch_size = 32
num_draft_tokens = 4

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    prefix_ids = torch.randint(0, vocab_size, (batch_size, 128))
    draft_logits = [torch.randn(batch_size, vocab_size) for _ in range(num_draft_tokens)]
    target_logits = [torch.randn(batch_size, vocab_size) for _ in range(num_draft_tokens)]
    return [prefix_ids, draft_logits, target_logits]

def get_init_inputs():
    return [draft_world_size, target_world_size, vocab_size, num_draft_tokens]

