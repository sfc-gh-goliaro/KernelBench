# KernelBench Level 6: Collective Communication Operators

This directory contains **20 collective communication operators** essential for distributed and parallel inference of large-scale AI models. These operators cover the communication primitives, parallelism strategies, and distributed computation patterns required to scale models across multiple GPUs and nodes.

## Overview

Level 6 operators are designed to benchmark:
- **Collective Primitives**: All-Reduce, All-Gather, Reduce-Scatter, All-to-All, Broadcast
- **Point-to-Point Communication**: Ring patterns, pipeline stages
- **Tensor Parallelism**: Column/Row parallel linear, distributed attention
- **Sequence Parallelism**: Ulysses, Ring Attention patterns
- **Pipeline Parallelism**: 1F1B scheduling, interleaved stages
- **Expert Parallelism**: MoE token routing across ranks
- **Memory Optimization**: ZeRO sharding, distributed KV cache
- **Inference Optimization**: Distributed sampling, speculative decoding

---

## Operator Categories

### 1. Collective Communication Primitives (1-6)

| # | Operator | Description | Use Cases |
|---|----------|-------------|-----------|
| 1 | `AllReduce` | Sum/Avg tensors across all ranks | Gradient sync, activation aggregation |
| 2 | `AllGather` | Gather tensors from all ranks | Tensor parallelism output gathering |
| 3 | `ReduceScatter` | Reduce then scatter to ranks | ZeRO optimizer, gradient sharding |
| 4 | `AllToAll` | Exchange data between all rank pairs | MoE expert routing |
| 5 | `Broadcast` | Send tensor from one rank to all | Parameter distribution |
| 6 | `PointToPoint` | Direct send/recv between ranks | Pipeline stages, ring attention |

### 2. Tensor Parallelism (7-8)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 7 | `TensorParallelLinear` | Column/Row parallel linear layers | Megatron-LM |
| 8 | `TensorParallelAttention` | Distributed attention heads | Megatron-LM |

### 3. Sequence & Context Parallelism (9, 12, 15)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 9 | `SequenceParallel` | Distribute sequence across ranks | Megatron-LM SP |
| 12 | `ContextParallel` | Long sequence distribution with ring KV | DeepSpeed, Ring Attention |
| 15 | `DistributedAttention` | Ulysses, Ring, Striped attention | Ulysses, Ring Attention |

### 4. Pipeline Parallelism (10)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 10 | `PipelineParallel` | 1F1B, interleaved, zero-bubble schedules | GPipe, PipeDream, ZeroBubble |

### 5. Expert Parallelism (11)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 11 | `ExpertParallel` | MoE token dispatch across ranks | GShard, DeepSeek-V3 |

### 6. KV Cache Distribution (13)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 13 | `KVCacheParallel` | Distributed/paged KV cache | vLLM, TensorRT-LLM |

### 7. Communication Optimization (14, 16)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 14 | `AsyncCommunication` | Overlapped compute-communication | NCCL async ops |
| 16 | `ZeROOptimizer` | Sharded optimizer states (Stages 1-3) | ZeRO, ZeRO++ |

### 8. Model Parallelism (17)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 17 | `ModelParallel` | 3D parallelism (TP + PP + DP) | Megatron-LM, Alpa |

### 9. Distributed Components (18-20)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 18 | `DistributedEmbedding` | Vocabulary/dimension partitioned embedding | Megatron-LM |
| 19 | `SpeculativeDecoding` | Distributed draft-verify coordination | SpecInfer |
| 20 | `DistributedSampling` | Top-k/Top-p across vocab partitions | vLLM |

---

## Model Coverage

These communication operators enable distributed inference for all models in Level 5:

### Large Language Models

| Model | Parallelism Strategy | Key Operators |
|-------|---------------------|---------------|
| **Mixtral-8x7B** | EP + TP | AllToAll, ExpertParallel, TensorParallelAttention |
| **DeepSeek-V3** | EP + TP + CP | ExpertParallel, ContextParallel, AllToAll |
| **Mistral-Large-3** | TP + PP | TensorParallelLinear, PipelineParallel |
| **MiniMax-M2** | EP + CP | ExpertParallel, ContextParallel, AsyncCommunication |
| **Ring-1T** | EP (1000+ experts) | ExpertParallel, AllToAll, ZeROOptimizer |
| **Kimi-K2** | EP + TP | ExpertParallel, TensorParallelAttention |

### Vision-Language Models

| Model | Parallelism Strategy | Key Operators |
|-------|---------------------|---------------|
| **Qwen3-VL-8B** | TP + CP | TensorParallel, ContextParallel, AllGather |
| **Qwen3-Omni-30B** | TP + EP | TensorParallel, ExpertParallel |
| **GLM-4.5V** | TP | TensorParallelLinear, TensorParallelAttention |
| **ERNIE-4.5-VL** | EP + TP | ExpertParallel, TensorParallel |

### Diffusion Models

| Model | Parallelism Strategy | Key Operators |
|-------|---------------------|---------------|
| **FLUX.1-dev** | TP | TensorParallelLinear, AllReduce |
| **StarFlow** | TP | TensorParallel, AllGather |

### Long-Context Models

| Model | Parallelism Strategy | Key Operators |
|-------|---------------------|---------------|
| **Kimi-Linear-48B** | CP + EP | ContextParallel, DistributedAttention, RingCommunication |
| **DeepSeek-OCR** | CP | ContextParallel, KVCacheParallel |

---

## Communication Patterns

### Ring Communication (for Ring Attention)
```
Rank 0 → Rank 1 → Rank 2 → ... → Rank N-1 → Rank 0
```
Used by: `ContextParallel`, `DistributedAttention (ring)`, `PointToPoint`

### All-to-All (for MoE)
```
All ranks exchange tokens with all other ranks based on expert assignments
```
Used by: `ExpertParallel`, `AllToAll`

### Hierarchical (for large-scale training)
```
Intra-node: Fast NVLink communication
Inter-node: Network communication  
```
Used by: `ZeROOptimizer (ZeRO++)`, `ModelParallel (3D)`

---

## Usage Example

```python
import torch
from KernelBench.level6 import AllReduce, TensorParallelLinear

# Initialize communication operators
world_size = 8
rank = 0  # Current rank

# All-Reduce for gradient synchronization
all_reduce = AllReduce(world_size=world_size, reduce_op='sum')
gradients = [torch.randn(1024, 1024) for _ in range(world_size)]
reduced = all_reduce(gradients)

# Tensor Parallel Linear
tp_linear = TensorParallelLinear(
    in_features=4096,
    out_features=4096, 
    world_size=world_size,
    rank=rank
)
x = torch.randn(32, 512, 4096)
output = tp_linear(x)
```

---

## Parallelism Strategies

### 1. Data Parallelism (DP)
- Each rank has full model, different data
- Uses: `AllReduce` for gradient sync

### 2. Tensor Parallelism (TP)
- Split model layers across ranks
- Uses: `TensorParallelLinear`, `TensorParallelAttention`, `AllReduce`, `AllGather`

### 3. Pipeline Parallelism (PP)
- Split model stages across ranks
- Uses: `PipelineParallel`, `PointToPoint`

### 4. Sequence Parallelism (SP)
- Split sequence across ranks
- Uses: `SequenceParallel`, `AllGather`, `ReduceScatter`

### 5. Context Parallelism (CP)
- Distribute long contexts with ring communication
- Uses: `ContextParallel`, `DistributedAttention`, `RingCommunication`

### 6. Expert Parallelism (EP)
- Distribute MoE experts across ranks
- Uses: `ExpertParallel`, `AllToAll`

### 7. 3D Parallelism (TP + PP + DP)
- Combines tensor, pipeline, and data parallelism
- Uses: `ModelParallel`, all collective primitives

---

## Performance Considerations

### Bandwidth-Optimal Algorithms
- **Ring All-Reduce**: O(2(N-1)/N × data_size) for N ranks
- **Ring All-Gather**: O((N-1)/N × data_size)
- **Tree Broadcast**: O(log(N) × latency)

### Latency-Optimal Algorithms
- **Tree All-Reduce**: O(log(N)) steps
- **Recursive Halving**: O(log(N)) for reduce-scatter

### Overlap Strategies
- **Async Communication**: Overlap compute with communication
- **Bucketed All-Reduce**: Pipeline gradient communication with backward pass
- **Chunked Ring**: Pipeline KV transfer with attention computation

---

## References

Key papers and systems:
- [Megatron-LM](https://arxiv.org/abs/1909.08053) - Tensor/Pipeline Parallelism
- [ZeRO](https://arxiv.org/abs/1910.02054) - Memory Optimization
- [GShard](https://arxiv.org/abs/2006.16668) - Expert Parallelism
- [Ring Attention](https://arxiv.org/abs/2310.01889) - Context Parallelism
- [Ulysses](https://arxiv.org/abs/2309.14509) - Sequence Parallelism
- [vLLM](https://arxiv.org/abs/2309.06180) - Paged Attention
- [SpecInfer](https://arxiv.org/abs/2305.09781) - Distributed Speculation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - ZeRO & Inference

---

## Integration with Level 5

Level 6 operators complement Level 5's emerging operators:

| Level 5 Operator | Level 6 Communication |
|-----------------|----------------------|
| MoELayer | ExpertParallel, AllToAll |
| MultiHeadLatentAttention | TensorParallelAttention, AllReduce |
| RingAttention | ContextParallel, PointToPoint |
| SharedExpertMoE | ExpertParallel, Broadcast |
| TrillionScaleMoE | ExpertParallel (hierarchical) |
| LightningAttention | DistributedAttention |
| FlashAttentionChunked | ContextParallel |

