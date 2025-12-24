# KernelBench Level 7: Fused Operator Blocks for LLM Serving

This directory contains **34 fused operator blocks** commonly used in LLM serving and training. These represent production-critical kernel fusions that eliminate memory traffic and enable compute-communication overlap.

## Overview

Level 7 operators are designed to benchmark:
- **Transformer Block Fusions**: RMSNorm + SwiGLU, RoPE + QKV, Attention blocks
- **SSM/RNN Fusions**: Mamba selective scan, RWKV linear attention
- **MoE Fusions**: Top-K gating + permutation, grouped GEMM
- **Quantization Fusions**: W4A16 dequant + GEMM, BitNet ternary GEMM
- **Communication Fusions**: Linear + All-Reduce, MoE All-to-All + GEMM
- **Attention Fusions**: Softmax + dropout + mask, KV cache update

---

## Operator Categories

### 1. Transformer Block Fusions (1-3, 12-15)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 1 | `FusedRMSNormSwiGLU` | RMSNorm + SwiGLU MLP in single pass | LLaMA, Mistral |
| 2 | `FusedRoPEQKVProjection` | RoPE + QKV projection (packed weights) | FlashAttention |
| 3 | `FusedLinearCrossEntropy` | LM head + cross entropy (Unsloth layer) | Unsloth, Cut-CE |
| 12 | `FusedResidualLayerNorm` | Residual add + LayerNorm/RMSNorm | Apex, FlashAttention |
| 13 | `FusedBiasActivation` | Bias + activation (GELU, SiLU, etc.) | cuBLAS epilogue |
| 14 | `FusedSoftmaxDropoutMask` | Softmax + dropout + mask application | FlashAttention |
| 15 | `FusedGeGLU` | SwiGLU/GeGLU/ReGLU with packed weights | GLU Variants |

### 2. SSM & Linear Attention Fusions (4-5, 18)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 4 | `MambaSelectiveScan` | Full Mamba S6 selective scan | Mamba, Mamba-2 |
| 5 | `FusedCausalConv1dActivation` | Causal Conv1d + SiLU for SSM | Mamba preprocessing |
| 18 | `FusedRWKVLinear` | RWKV time-mixing + WKV computation | RWKV-v5/v6 |

### 3. MoE Fusions (6-7, 17)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 6 | `FusedTopKGatingPermutation` | Top-K routing + token permutation | Megablocks |
| 7 | `GroupedGEMM` | Batched expert GEMM with variable sizes | Megablocks, Triton |
| 17 | `FusedMoEAllToAllGEMM` | All-to-All dispatch + expert GEMM | DeepSpeed-MoE |

### 4. Quantization Fusions (8-9)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 8 | `FusedW4A16DequantGEMM` | Block-wise W4A16 dequant + GEMM | GPTQ, AWQ, Marlin |
| 9 | `FusedBitNetGEMM` | BitNet 1.58-bit ternary GEMM | BitNet b1.58 |

### 5. Vision & Multimodal Fusions (10)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 10 | `FusedPatchEmbedLayerNorm` | Patch embedding + LayerNorm | ViT, SigLIP |

### 6. Distributed/Attention Fusions (11, 16, 19-20, 27-28)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 11 | `FusedBlockDiagonalAttention` | Block-diagonal masked attention | Ring Attention |
| 16 | `FusedLinearAllReduce` | Row-parallel linear + All-Reduce | Megatron-LM |
| 19 | `FusedAttentionQKVProj` | Full attention: QKV + attn + output | FlashAttention |
| 20 | `FusedKVCacheUpdate` | KV cache update + attention | vLLM, FlashInfer |
| 27 | `FusedGEMMReduceScatter` | GEMM + Reduce-Scatter for TP | Megatron-LM, DeepSpeed |
| 28 | `FusedAllGatherGEMM` | All-Gather + GEMM for TP | Megatron-LM, DeepSpeed |

### 7. AllReduce + Normalization Fusions (21-22)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 21 | `FusedAllReduceRMSNorm` | AllReduce + RMSNorm | vLLM, SGLang |
| 22 | `FusedAllReduceRMSNormQuant` | AllReduce + RMSNorm + FP8 Quant | vLLM FP8, SGLang |

### 8. Quantization Fusions (23-26)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 23 | `FusedAttentionOutputQuant` | Attention + FP8 output quantization | vLLM FP8, TRT-LLM |
| 24 | `FusedSiLUQuant` | SiLU + FP8/INT8 quantization | vLLM, Transformer Engine |
| 25 | `FusedRMSNormFP8Quant` | RMSNorm + FP8 quantization | vLLM, SGLang |
| 26 | `FusedMoEQuantized` | MoE with FP8/INT8 quantization | vLLM FP8 MoE |

### 9. MoE + Activation Fusions (29)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 29 | `FusedMoESiLUMul` | MoE + SwiGLU (SiLU * Mul) fusion | Mixtral, vLLM |

### 10. Add + Normalization (30)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 30 | `FusedAddRMSNorm` | Residual Add + RMSNorm (fused_add_rms_norm) | vLLM, SGLang |

### 11. SGLang-Specific Fusions (31-34)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 31 | `FusedMatmulSwiGLU` | Matmul fused with SiLU-and-mul (fused_swiglu) | SGLang, Triton |
| 32 | `FusedW8A8GEMM` | INT8 GEMM with W8A8 quantization | SGLang, SmoothQuant |
| 33 | `FusedQuantizedKVCacheAttention` | Attention with 4/8-bit quantized KV cache + dequant | SGLang, KIVI |
| 34 | `FusedMoEDeepGEMM` | DeepGEMM-optimized MoE with constant latency | SGLang DeepGEMM |

---

## Key Fusions Explained

### 1. Fused RMSNorm + SwiGLU
```
Standard:     RMSNorm(x) → store → Gate(x) → store → Up(x) → store → SiLU(gate)*up → store → Down
Fused:        x → [RMSNorm → Gate/Up → SiLU*multiply] → store → Down
Memory saved: 3 intermediate tensors
```

### 2. Fused Linear + Cross Entropy (Unsloth)
```
Standard:     hidden → LM_head (store 128K logits) → CrossEntropy
Fused:        hidden → [chunk-wise logits → online softmax → loss] → scalar
Memory saved: O(batch × seq × vocab) → O(batch × seq)
```

### 3. Grouped GEMM for MoE
```
Standard:     for expert in experts: output[expert] = expert(tokens[expert])
Fused:        outputs = grouped_gemm(all_tokens, all_weights, group_sizes)
Speedup:      Single kernel launch, vectorized loads
```

### 4. Communication-Computation Overlap
```
Standard:     Compute → Wait → Communicate → Wait → Next Compute
Fused:        Compute_chunk_1 → [Communicate_chunk_0 || Compute_chunk_2] → ...
```

---

## Integration with Other Levels

| Level 5 Operator | Level 7 Fused Version |
|-----------------|----------------------|
| SwiGLU (53) | FusedRMSNormSwiGLU (1) |
| RoPE (28) | FusedRoPEQKVProjection (2) |
| S4StateSpace (7) | MambaSelectiveScan (4) |
| TopKGating (9) | FusedTopKGatingPermutation (6) |
| W4A16Matmul (23) | FusedW4A16DequantGEMM (8) |
| BitLinear (41) | FusedBitNetGEMM (9) |

| Level 6 Operator | Level 7 Fused Version |
|-----------------|----------------------|
| AllReduce (1) | FusedLinearAllReduce (16) |
| AllToAll (4) | FusedMoEAllToAllGEMM (17) |
| KVCacheParallel (13) | FusedKVCacheUpdate (20) |

---

## Usage Example

```python
import torch
from KernelBench.level7 import FusedRMSNormSwiGLU

# Initialize fused layer
fused_mlp = FusedRMSNormSwiGLU(
    hidden_dim=4096,
    intermediate_dim=11008,
    eps=1e-6
)

# Forward pass
x = torch.randn(32, 2048, 4096)
output = fused_mlp(x)

# Equivalent to:
# x = RMSNorm(x)
# gate = gate_proj(x)
# up = up_proj(x)
# x = silu(gate) * up
# output = down_proj(x)
```

---

## Performance Characteristics

### Memory Savings

| Fusion | Memory Reduction |
|--------|-----------------|
| RMSNorm + SwiGLU | ~3x intermediate tensors |
| Linear + CrossEntropy | O(vocab) → O(1) per token |
| Softmax + Dropout + Mask | ~2x (no stored attn scores) |
| W4A16 Dequant + GEMM | No dequantized weight storage |

### Compute-Communication Overlap

| Fusion | Overlap Strategy |
|--------|-----------------|
| Linear + AllReduce | Chunked matmul with pipelined reduce |
| MoE All-to-All + GEMM | Dispatch while computing previous batch |
| KV Cache + Attention | Update cache during attention compute |

---

## Supported Models

These fusions accelerate inference for:

| Model Family | Key Fusions Used |
|-------------|------------------|
| **LLaMA/Mistral** | 1, 2, 12, 13, 14, 15, 19, 20 |
| **Mamba/State Space** | 4, 5, 18 |
| **Mixtral/DeepSeek-MoE** | 1, 6, 7, 16, 17 |
| **RWKV** | 5, 18 |
| **BitNet** | 9, 12 |
| **Quantized (GPTQ/AWQ)** | 8, 20 |
| **Vision-Language** | 10, 11, 19 |

---

## Kernel Characteristics

### Memory-Bound Fusions
- FusedRMSNormSwiGLU
- FusedResidualLayerNorm
- FusedBiasActivation

### Compute-Bound Fusions
- GroupedGEMM
- FusedW4A16DequantGEMM
- FusedAttentionQKVProj

### Communication-Bound Fusions
- FusedLinearAllReduce
- FusedMoEAllToAllGEMM

---

## References

Key papers and implementations:
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [Megablocks](https://arxiv.org/abs/2211.15841) - Grouped GEMM for MoE
- [Mamba](https://arxiv.org/abs/2312.00752) - Selective scan
- [RWKV](https://arxiv.org/abs/2305.13048) - Linear attention
- [BitNet](https://arxiv.org/abs/2310.11453) - Ternary quantization
- [Unsloth](https://github.com/unslothai/unsloth) - Fused cross entropy
- [Marlin](https://github.com/IST-DASLab/marlin) - W4A16 kernels
- [vLLM](https://arxiv.org/abs/2309.06180) - Paged attention

