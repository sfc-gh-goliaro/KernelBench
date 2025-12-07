# KernelBench Level 5: Emerging Operators

This directory contains **70 emerging operators** representing cutting-edge components from modern AI architectures. These operators cover the latest developments in efficient attention, mixture-of-experts, multimodal processing, quantization, and more.

## Overview

Level 5 operators are designed to benchmark:
- **Linear Attention Variants**: O(N) complexity alternatives to standard attention
- **Mixture-of-Experts (MoE)**: Sparse computation for trillion-parameter models  
- **Speculative Decoding**: Accelerated autoregressive generation
- **Multimodal Processing**: Vision-language, audio-visual fusion
- **Quantization Kernels**: W4A16, FP8, FP4, BF16 optimizations
- **Advanced Position Encodings**: RoPE, ALiBi, 2D/3D variants
- **Long Context Mechanisms**: Sliding window, paged attention, ring attention

---

## Operator Categories

### 1. Linear Attention Variants (1-8)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 1 | `LinearAttention` | Basic O(N) linear attention with kernel feature maps | Transformers are RNNs |
| 2 | `RWKVLinearAttention` | RWKV time-mixing attention mechanism | RWKV |
| 3 | `RetNetRetention` | Retention mechanism with exponential decay | RetNet |
| 4 | `GatedLinearAttention` | GLA with data-dependent gating | GLA |
| 5 | `BasedLinearAttention` | Taylor expansion-based linear attention | Based |
| 6 | `HyenaOperator` | O(N log N) long convolution + gating | Hyena |
| 7 | `S4StateSpace` | Structured state space with HiPPO | S4 |
| 8 | `H3StateSpace` | H3 combining SSM with multiplicative gating | H3 |

### 2. Mixture-of-Experts Components (9-13, 59-60, 68)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 9 | `TopKGating` | Top-k expert routing with load balancing | GShard |
| 10 | `ExpertParallelDispatch` | Token dispatch to experts with capacity | GShard |
| 11 | `SwitchRouting` | Switch Transformer top-1 routing | Switch Transformer |
| 12 | `ExpertChoiceRouting` | Experts choose tokens (perfect balance) | Expert Choice |
| 13 | `MoELayer` | Complete MoE layer with auxiliary loss | MoE |
| 59 | `MoEAttention` | MoE applied to attention heads | MoE Attention |
| 60 | `SharedExpertMoE` | DeepSeek-style shared + routed experts | DeepSeek-V2 |
| 68 | `TrillionScaleMoE` | Large-scale MoE (100+ experts) | Ring-1T |

### 3. Speculative Decoding (14-17)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 14 | `TreeAttention` | Tree-structured speculation attention | SpecInfer |
| 15 | `SpeculativeVerification` | Draft token verification logic | Speculative Decoding |
| 16 | `MedusaHeads` | Multiple parallel draft heads | Medusa |
| 17 | `EAGLEDrafting` | EAGLE feature prediction drafting | EAGLE |

### 4. Multimodal Operators (18-22, 38, 43, 56-57, 62)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 18 | `FlexiblePatchEmbedding` | ViT patch embedding with overlap support | ViT |
| 19 | `CrossAttentionFusion` | Flamingo-style cross-modal attention | Flamingo |
| 20 | `PerceiverCrossAttention` | Perceiver with learned latents | Perceiver |
| 21 | `CLIPContrastive` | CLIP contrastive learning module | CLIP |
| 22 | `QFormer` | Q-Former for vision-language alignment | BLIP-2 |
| 38 | `FeaturePyramidFusion` | FPN for multi-scale features | FPN |
| 43 | `VisionLanguageEmbedding` | VLM embedding fusion layer | LLaVA |
| 56 | `DynamicResolutionViT` | Dynamic resolution vision encoder | NaViT |
| 57 | `AudioVisualFusion` | Audio-visual multimodal fusion | Qwen-Omni |
| 62 | `NaViTPacking` | NaViT-style variable resolution packing | NaViT |

### 5. Quantization Kernels (23-27, 55, 69)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 23 | `W4A16Matmul` | 4-bit weights, 16-bit activations | GPTQ |
| 24 | `FP8Matmul` | FP8 (e4m3/e5m2) quantized matmul | FP8 Formats |
| 25 | `DynamicQuantization` | Per-token dynamic INT8 quantization | LLM.int8() |
| 26 | `SmoothQuantLinear` | SmoothQuant migration approach | SmoothQuant |
| 27 | `AWQGemm` | Activation-aware weight quantization | AWQ |
| 55 | `FP4Quantization` | NF4/FP4 quantization (QLoRA) | QLoRA |
| 69 | `BF16MatMul` | BF16 precision matrix multiplication | Mixed Precision |

### 6. Position Encodings (28-29, 39, 54, 61)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 28 | `RotaryPositionalEmbedding` | RoPE with extended context support | RoFormer |
| 29 | `ALiBi` | Attention with Linear Biases | ALiBi |
| 39 | `LongRoPE` | Extended context RoPE (2M+ tokens) | LongRoPE |
| 54 | `RoPE2D` | 2D RoPE for vision transformers | Vision RoPE |
| 61 | `RoPE3D` | 3D RoPE for video/spatiotemporal | Video Transformers |

### 7. Attention Variants (30-37, 40, 42, 58, 63-65)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 30 | `GroupedQueryAttention` | GQA with shared KV heads | GQA |
| 31 | `MultiQueryAttention` | MQA with single KV head | MQA |
| 32 | `SlidingWindowAttention` | Sliding window (Mistral-style) | Longformer/Mistral |
| 33 | `PagedAttention` | vLLM-style paged KV cache | vLLM |
| 34 | `KVCacheUpdate` | KV cache management operations | Inference Optimization |
| 35 | `FlashAttentionChunked` | Flash attention tiled computation | FlashAttention |
| 36 | `RingAttention` | Ring attention for distributed context | Ring Attention |
| 37 | `ParallelContextWindow` | PCW for extended context | PCW |
| 40 | `DifferentialAttention` | Diff attention (noise reduction) | Differential Transformer |
| 42 | `StripminingAttention` | Memory-efficient tiled attention | Memory Optimization |
| 58 | `EncoderDecoderCrossAttention` | Enc-Dec cross attention | Transformers |
| 63 | `CrossEncoderReranker` | Cross-encoder for retrieval reranking | Jina Reranker |
| 64 | `LightningAttention` | O(N) lightning attention | Lightning Attention-2 |
| 65 | `GLMPrefixLM` | GLM-style prefix language model | GLM-4 |

### 8. Diffusion & Flow Models (46-48, 52)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 46 | `AdaptiveLayerNorm` | AdaLN/AdaLN-Zero for diffusion | DiT |
| 47 | `TimestepEmbedding` | Timestep encoding for diffusion | DDPM |
| 48 | `DiTBlock` | DiT transformer block | DiT/FLUX |
| 52 | `FlowMatching` | Flow matching for generative models | Rectified Flow |

### 9. Audio Processing (49-50)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 49 | `MelSpectrogram` | Mel spectrogram computation | Whisper |
| 50 | `WhisperAudioEncoder` | Whisper-style audio encoder | Whisper |

### 10. OCR & Document Understanding (66-67, 70)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 66 | `DocumentLayoutAnalysis` | DETR-style layout detection | LayoutLM |
| 67 | `CTCDecoder` | CTC decoder for text recognition | CTC |
| 70 | `TextDetection` | DBNet-style text detection | DBNet |

### 11. Additional Operators (41, 44-45, 51, 53)

| # | Operator | Description | Reference |
|---|----------|-------------|-----------|
| 41 | `BitLinear` | 1-bit weight quantization (BitNet) | BitNet |
| 44 | `MomentumDistillation` | DINO/MoCo self-distillation | DINO |
| 45 | `SparseFFN` | Activation-sparse feed-forward | Deja Vu |
| 51 | `MultiHeadLatentAttention` | DeepSeek MLA (KV compression) | DeepSeek-V2 |
| 53 | `SwiGLU` | SwiGLU/GeGLU/ReGLU activations | GLU Variants |

---

## Supported Hugging Face Models

The operators in Level 5 provide full coverage for the following models:

### Language Models

| Model | Key Operators Used |
|-------|-------------------|
| **mistralai/Mixtral-8x7B-Instruct-v0.1** | MoELayer, TopKGating, SlidingWindowAttention, SwiGLU, RoPE |
| **deepseek-ai/DeepSeek-V3** | MultiHeadLatentAttention, SharedExpertMoE, RoPE |
| **microsoft/phi-2** | RoPE, SwiGLU, Standard Attention |
| **google/gemma-7b** | RoPE, GQA, SwiGLU |
| **mistralai/Mistral-Large-3** | GQA, SlidingWindowAttention, SwiGLU, RoPE |
| **zai-org/GLM-4.5** | GLMPrefixLM, GQA, RoPE |
| **internlm/Intern-S1** | RoPE, GQA, SwiGLU |
| **ByteDance-Seed/Seed-OSS-36B-Instruct** | GQA, RoPE, SwiGLU |
| **MiniMaxAI/MiniMax-M2** | LightningAttention, MoELayer, ParallelContextWindow |
| **moonshotai/Kimi-K2-Thinking** | MoE, Linear Attention variants |
| **moonshotai/Kimi-Linear-48B** | GatedLinearAttention, MoELayer |
| **inclusionAI/Ring-1T-FP8** | TrillionScaleMoE, FP8Matmul |

### Vision-Language Models

| Model | Key Operators Used |
|-------|-------------------|
| **Qwen/Qwen3-VL-8B-Instruct** | DynamicResolutionViT, CrossAttentionFusion, RoPE2D |
| **Qwen/Qwen3-Omni-30B** | AudioVisualFusion, DynamicResolutionViT |
| **Qwen/Qwen-Image-Edit** | VisionLanguageEmbedding, RoPE2D |
| **zai-org/GLM-4.5V** | GLMPrefixLM, CrossAttentionFusion, DynamicResolutionViT |
| **baidu/ERNIE-4.5-VL-28B-A3B-PT** | NaViTPacking, RoPE3D, MoE |
| **nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16** | BF16MatMul, VisionLanguageEmbedding, CrossAttentionFusion |
| **deepseek-ai/DeepSeek-OCR** | DynamicResolutionViT, CrossAttentionFusion, MLA |

### Image Generation Models

| Model | Key Operators Used |
|-------|-------------------|
| **black-forest-labs/FLUX.1-dev** | DiTBlock, AdaptiveLayerNorm, TimestepEmbedding, RoPE2D |
| **apple/starflow** | FlowMatching, LinearAttention |

### Audio Models

| Model | Key Operators Used |
|-------|-------------------|
| **openai/whisper-large-v3** | MelSpectrogram, WhisperAudioEncoder, EncoderDecoderCrossAttention |

### OCR & Document Models

| Model | Key Operators Used |
|-------|-------------------|
| **tencent/HunyuanOCR** | TextDetection, DocumentLayoutAnalysis, CTCDecoder |
| **PaddlePaddle/PaddleOCR-VL** | NaViTPacking, RoPE3D, CTCDecoder, TextDetection |
| **zai-org/Glyph** | TextDetection, CTCDecoder |

### Retrieval & Reranking

| Model | Key Operators Used |
|-------|-------------------|
| **jinaai/jina-reranker-m0** | CrossEncoderReranker |

### Quantized Models

| Model | Key Operators Used |
|-------|-------------------|
| **unsloth/Mistral-Large-3-675B-Instruct-2512-NVFP4** | FP4Quantization, FP8Matmul |
| **unsloth/Ministral-3-14B-Instruct-2512-GGUF** | W4A16Matmul, DynamicQuantization |

---

## Usage

Each operator follows the KernelBench standard format:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """Operator description and reference."""
    
    def __init__(self, ...):
        """Initialize parameters."""
        super(Model, self).__init__()
        # ... initialization
    
    def forward(self, x, ...):
        """Forward pass."""
        # ... computation
        return output

# Test configuration
def get_inputs():
    """Return sample inputs for benchmarking."""
    return [torch.randn(...)]

def get_init_inputs():
    """Return initialization parameters."""
    return [param1, param2, ...]
```

### Running Benchmarks

```python
from KernelBench.level5 import LinearAttention

# Load operator
spec = importlib.util.spec_from_file_location("module", "KernelBench/level5/1_LinearAttention.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Initialize model
init_args = module.get_init_inputs()
model = module.Model(*init_args).cuda()

# Get sample inputs
inputs = [x.cuda() for x in module.get_inputs()]

# Run forward pass
output = model(*inputs)
```

---

## Contributing

When adding new operators:

1. Follow the naming convention: `{number}_{OperatorName}.py`
2. Include comprehensive docstrings with paper references
3. Implement `Model`, `get_inputs()`, and `get_init_inputs()`
4. Use realistic test parameters for benchmarking
5. Ensure syntax validation passes

---

## References

Key papers and resources:
- [Transformers are RNNs](https://arxiv.org/abs/2006.16236) - Linear Attention
- [RWKV](https://arxiv.org/abs/2305.13048) - RWKV Architecture
- [RetNet](https://arxiv.org/abs/2307.08621) - Retentive Network
- [Mamba](https://arxiv.org/abs/2312.00752) - State Space Models
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - MoE
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - MLA & Shared Experts
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Memory-Efficient Attention
- [CLIP](https://arxiv.org/abs/2103.00020) - Contrastive Learning
- [DiT](https://arxiv.org/abs/2212.09748) - Diffusion Transformers
- [Whisper](https://arxiv.org/abs/2212.04356) - Speech Recognition
- [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient Fine-tuning

