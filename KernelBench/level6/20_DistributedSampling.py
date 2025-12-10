import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


TASK_CONFIG = {
    'comparison_mode': 'default',
    'atol': 1e-4,
    'rtol': 1e-4,
}

class Model(nn.Module):
    """
    Distributed Token Sampling for Parallel Generation.
    
    Coordinates sampling across distributed vocabulary partitions.
    Handles top-k, top-p, and temperature sampling with proper
    synchronization across ranks.
    
    Essential for distributed inference with vocabulary parallelism.
    """
    def __init__(self, vocab_size, world_size, rank, 
                 temperature=1.0, top_k=50, top_p=0.95):
        """
        :param vocab_size: Total vocabulary size
        :param world_size: Number of ranks
        :param rank: Current rank
        :param temperature: Sampling temperature
        :param top_k: Top-k filtering
        :param top_p: Nucleus sampling threshold
        """
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Vocabulary partition
        assert vocab_size % world_size == 0
        self.vocab_per_rank = vocab_size // world_size
        self.vocab_start = rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
    
    def local_top_k(self, local_logits):
        """
        Compute local top-k candidates.
        
        :param local_logits: Logits for this rank's vocabulary partition
        :return: (top_k values, top_k indices, global indices)
        """
        # Apply temperature
        scaled_logits = local_logits / self.temperature
        
        # Local top-k
        k = min(self.top_k, scaled_logits.shape[-1])
        values, indices = torch.topk(scaled_logits, k, dim=-1)
        
        # Convert to global indices
        global_indices = indices + self.vocab_start
        
        return values, indices, global_indices
    
    def distributed_top_k(self, local_topk_values, local_topk_indices, 
                          all_rank_topk=None):
        """
        Combine local top-k across ranks to find global top-k.
        
        :param local_topk_values: This rank's top-k values
        :param local_topk_indices: This rank's top-k global indices
        :param all_rank_topk: All ranks' top-k (for simulation)
        :return: Global top-k values and indices
        """
        if all_rank_topk is None:
            return local_topk_values, local_topk_indices
        
        # Concatenate all ranks' top-k
        all_values = torch.cat([t[0] for t in all_rank_topk], dim=-1)
        all_indices = torch.cat([t[1] for t in all_rank_topk], dim=-1)
        
        # Global top-k
        k = min(self.top_k, all_values.shape[-1])
        global_values, selection = torch.topk(all_values, k, dim=-1)
        global_indices = torch.gather(all_indices, -1, selection)
        
        return global_values, global_indices
    
    def top_p_filter(self, logits, sorted_indices):
        """
        Apply nucleus (top-p) filtering.
        
        :param logits: Sorted logits
        :param sorted_indices: Corresponding indices
        :return: Filtered logits and indices
        """
        probs = F.softmax(logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)
        
        # Find cutoff
        mask = cumsum_probs <= self.top_p
        mask[..., 0] = True  # Always keep at least one
        
        # Filter
        filtered_logits = logits.masked_fill(~mask, float('-inf'))
        
        return filtered_logits
    
    def sample(self, logits):
        """
        Sample from filtered distribution.
        
        :param logits: Filtered logits
        :return: Sampled token index
        """
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)
    
    def forward(self, local_logits, all_rank_logits=None, 
                use_top_p=True, use_top_k=True):
        """
        Distributed sampling forward pass.
        
        :param local_logits: Logits for local vocabulary partition
        :param all_rank_logits: Logits from all ranks (for simulation)
        :param use_top_p: Apply nucleus sampling
        :param use_top_k: Apply top-k filtering
        :return: Sampled token IDs
        """
        batch_size = local_logits.shape[0]
        
        # Get local top-k
        local_values, local_indices, global_indices = self.local_top_k(local_logits)
        
        if all_rank_logits is not None:
            # Simulate gathering top-k from all ranks
            all_topk = []
            for r, logits in enumerate(all_rank_logits):
                values = logits / self.temperature
                k = min(self.top_k, values.shape[-1])
                v, i = torch.topk(values, k, dim=-1)
                global_i = i + r * self.vocab_per_rank
                all_topk.append((v, global_i))
            
            # Global top-k
            global_values, global_indices = self.distributed_top_k(
                local_values, global_indices, all_topk
            )
        else:
            global_values = local_values
        
        # Apply top-p filtering
        if use_top_p:
            sorted_logits, sort_indices = torch.sort(global_values, descending=True, dim=-1)
            filtered_logits = self.top_p_filter(sorted_logits, sort_indices)
            
            # Unsort
            unsort_indices = torch.argsort(sort_indices, dim=-1)
            global_values = torch.gather(filtered_logits, -1, unsort_indices)
        
        # Sample
        sampled_local_idx = self.sample(global_values)
        
        # Convert to global token ID
        sampled_token = torch.gather(global_indices, -1, sampled_local_idx.unsqueeze(-1))
        
        return sampled_token.squeeze(-1)


# Beam Search with Distributed Vocabulary
class DistributedBeamSearch(nn.Module):
    """
    Distributed beam search across vocabulary partitions.
    """
    def __init__(self, vocab_size, world_size, rank, num_beams=4, max_length=100):
        super(DistributedBeamSearch, self).__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank
        self.num_beams = num_beams
        self.max_length = max_length
        
        self.vocab_per_rank = vocab_size // world_size
        self.vocab_start = rank * self.vocab_per_rank
    
    def beam_step(self, local_scores, beam_indices, all_rank_scores=None):
        """
        Single beam search step with distributed vocabulary.
        
        :param local_scores: Scores for local vocabulary
        :param beam_indices: Current beam token sequences
        :param all_rank_scores: Scores from all ranks
        :return: Updated beams
        """
        batch_size, num_beams, local_vocab = local_scores.shape
        
        if all_rank_scores is not None:
            # Gather all scores
            all_scores = torch.cat(all_rank_scores, dim=-1)
        else:
            all_scores = local_scores
        
        # Reshape for beam selection: (batch, num_beams * vocab)
        all_scores = all_scores.view(batch_size, -1)
        
        # Top-k for new beams
        topk_scores, topk_indices = torch.topk(all_scores, self.num_beams, dim=-1)
        
        # Convert to beam and token indices
        new_beam_indices = topk_indices // self.vocab_size
        new_token_indices = topk_indices % self.vocab_size
        
        return topk_scores, new_beam_indices, new_token_indices
    
    def forward(self, initial_logits, all_rank_logits=None):
        """
        Distributed beam search.
        """
        batch_size = initial_logits.shape[0]
        
        # Initialize beams
        scores = F.log_softmax(initial_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.num_beams, dim=-1)
        
        return topk_indices


# Test parameters
vocab_size = 128000
world_size = 8
rank = 0
batch_size = 32

def get_inputs(**kwargs):
    """
    Generate inputs for the model.
    
    Args:
        **kwargs: Override default dimensions (e.g., batch_size=32)
    
    Returns:
        List of input tensors
    """
    # Local logits (for vocabulary partition)
    local_vocab = vocab_size // world_size
    local_logits = torch.randn(batch_size, local_vocab)
    return [local_logits]

def get_init_inputs():
    return [vocab_size, world_size, rank]

