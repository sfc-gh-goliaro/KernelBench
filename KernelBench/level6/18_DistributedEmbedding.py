import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Distributed Embedding Table.
    
    Partitions large embedding tables across ranks for vocabulary
    or mixture-of-expert embeddings. Essential for LLMs with
    large vocabularies.
    
    Based on: Megatron-LM parallel embedding
    """
    def __init__(self, num_embeddings, embedding_dim, world_size, rank,
                 padding_idx=None, partition='vocab'):
        """
        :param num_embeddings: Total vocabulary size
        :param embedding_dim: Embedding dimension
        :param world_size: Number of ranks
        :param rank: Current rank
        :param padding_idx: Padding index
        :param partition: 'vocab' for vocabulary partition, 'dim' for dimension partition
        """
        super(Model, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank
        self.partition = partition
        self.padding_idx = padding_idx
        
        if partition == 'vocab':
            # Partition vocabulary across ranks
            assert num_embeddings % world_size == 0
            self.vocab_per_rank = num_embeddings // world_size
            self.vocab_start = rank * self.vocab_per_rank
            self.vocab_end = self.vocab_start + self.vocab_per_rank
            self.local_embedding = nn.Embedding(
                self.vocab_per_rank, embedding_dim,
                padding_idx=padding_idx - self.vocab_start 
                    if padding_idx is not None and 
                       self.vocab_start <= padding_idx < self.vocab_end 
                    else None
            )
        else:  # partition == 'dim'
            # Partition embedding dimension across ranks
            assert embedding_dim % world_size == 0
            self.dim_per_rank = embedding_dim // world_size
            self.local_embedding = nn.Embedding(
                num_embeddings, self.dim_per_rank, padding_idx=padding_idx
            )
    
    def forward(self, input_ids, all_rank_outputs=None):
        """
        Distributed embedding lookup.
        
        :param input_ids: Token IDs (batch, seq)
        :param all_rank_outputs: For all-reduce simulation
        :return: Embeddings
        """
        if self.partition == 'vocab':
            # Check which tokens are in this rank's partition
            mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
            
            # Local lookup
            local_ids = torch.where(
                mask,
                input_ids - self.vocab_start,
                torch.zeros_like(input_ids)
            )
            local_embeddings = self.local_embedding(local_ids)
            
            # Zero out embeddings for tokens not in this partition
            local_embeddings = local_embeddings * mask.unsqueeze(-1).float()
            
            if all_rank_outputs is not None:
                # All-reduce to combine partial embeddings
                all_rank_outputs.append(local_embeddings)
                if len(all_rank_outputs) == self.world_size:
                    return sum(all_rank_outputs)
            
            return local_embeddings
        
        else:  # dimension partition
            # Each rank has full vocab but partial embedding dim
            local_embeddings = self.local_embedding(input_ids)
            
            if all_rank_outputs is not None:
                # All-gather along dimension
                all_rank_outputs.append(local_embeddings)
                if len(all_rank_outputs) == self.world_size:
                    return torch.cat(all_rank_outputs, dim=-1)
            
            return local_embeddings


# Parallel Output Embedding (tied with input embedding)
class ParallelOutputEmbedding(nn.Module):
    """
    Parallel output embedding for language model head.
    
    Computes logits distributed across vocabulary partition.
    """
    def __init__(self, embedding_dim, num_embeddings, world_size, rank):
        super(ParallelOutputEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.world_size = world_size
        self.rank = rank
        
        # Vocabulary partition
        assert num_embeddings % world_size == 0
        self.vocab_per_rank = num_embeddings // world_size
        
        # Local output weight (can be tied to input embedding)
        self.weight = nn.Parameter(
            torch.randn(self.vocab_per_rank, embedding_dim) * 0.02
        )
    
    def forward(self, hidden_states, all_rank_outputs=None):
        """
        Compute local logits and optionally gather.
        
        :param hidden_states: (batch, seq, embedding_dim)
        :param all_rank_outputs: For gathering all logits
        :return: Local logits (batch, seq, vocab_per_rank)
        """
        # Local matmul
        local_logits = F.linear(hidden_states, self.weight)
        
        if all_rank_outputs is not None:
            # All-gather logits
            all_rank_outputs.append(local_logits)
            if len(all_rank_outputs) == self.world_size:
                return torch.cat(all_rank_outputs, dim=-1)
        
        return local_logits


# Mixture-of-Experts Embedding
class MoEEmbedding(nn.Module):
    """
    Distributed embedding with expert routing.
    
    Different embedding experts for different token types.
    """
    def __init__(self, num_embeddings, embedding_dim, num_experts, 
                 world_size, rank):
        super(MoEEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        
        # Each rank has subset of experts
        assert num_experts % world_size == 0
        self.experts_per_rank = num_experts // world_size
        
        # Expert embeddings
        self.expert_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(self.experts_per_rank)
        ])
        
        # Router
        self.router = nn.Linear(embedding_dim, num_experts)
    
    def forward(self, input_ids, routing_probs=None):
        """
        MoE embedding lookup.
        
        :param input_ids: Token IDs
        :param routing_probs: Pre-computed routing (optional)
        :return: Routed embeddings
        """
        batch, seq = input_ids.shape
        
        # Get all expert embeddings for this rank
        expert_outputs = []
        for expert in self.expert_embeddings:
            expert_outputs.append(expert(input_ids))
        
        # Stack: (num_experts_per_rank, batch, seq, dim)
        expert_stack = torch.stack(expert_outputs, dim=0)
        
        # Simple averaging (full routing would need all-to-all)
        output = expert_stack.mean(dim=0)
        
        return output


# Test parameters
num_embeddings = 128000  # Large vocabulary
embedding_dim = 4096
world_size = 8
rank = 0
batch_size = 32
seq_len = 512

def get_inputs():
    # Token IDs covering full vocabulary
    input_ids = torch.randint(0, num_embeddings, (batch_size, seq_len))
    return [input_ids]

def get_init_inputs():
    return [num_embeddings, embedding_dim, world_size, rank]

