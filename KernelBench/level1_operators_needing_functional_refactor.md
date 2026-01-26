# Level1 Operators Needing Functional Refactor

This document tracks level1 operators that need to be refactored to become functional operators (like PyTorch's functional operators). Functional operators should not have any learnable parameters or stored state; instead, all parameters should be passed at call time.

## Summary

- **Total level1 operators used in level4:** 21
- **Already functional (no changes needed):** 8
- **Need refactoring:** 13

---

## Operators Already Functional (No Changes Needed)

These operators have empty `__init__` methods (or only call `super().__init__()`) and no stored state:

| # | File Path | Operator Name | Notes |
|---|-----------|---------------|-------|
| 1 | `activations/_1_ReLU.py` | ReLU | Uses `torch.relu(x)` directly |
| 2 | `activations/_3_Sigmoid.py` | Sigmoid | Uses `torch.sigmoid(x)` directly |
| 3 | `activations/_5_Softmax.py` | Softmax | Uses `torch.softmax(x, dim=1)` directly |
| 4 | `activations/_7_Swish.py` | Swish | Uses `x * torch.sigmoid(x)` directly |
| 5 | `activations/_8_GELU.py` | GELU | Uses `F.gelu(x)` directly |
| 6 | `activations/_11_Softplus.py` | Softplus | Uses `F.softplus(x)` directly |
| 7 | `matmul/_1_MatMul.py` | MatMul | Uses `torch.matmul(A, B)` directly |
| 8 | `pooling/_9_MeanPooling.py` | MeanPooling | Uses `hidden_states.mean(dim=1)` directly |

---

## Operators Needing Refactoring

These operators have stored state (config params, `nn.Parameter`, `register_buffer`, or wrap `nn.Module` subclasses with learnable parameters):

### 1. Normalization Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 1 | `normalization/_1_BatchNorm.py` | BatchNorm | Wraps `nn.BatchNorm2d` with learnable weight/bias and running stats | Pass weight, bias, running_mean, running_var as parameters |
| 2 | `normalization/_3_GroupNorm.py` | GroupNorm | Wraps `nn.GroupNorm` with learnable weight/bias | Pass weight, bias, num_groups, num_channels as parameters |
| 3 | `normalization/_4_RMSNorm.py` | RMSNorm | Stores `num_features`, `eps` as instance attributes | Pass `eps` as parameter; level4 wrapper adds learnable `weight` |
| 4 | `normalization/_6_LayerNorm.py` | LayerNorm | Wraps `nn.LayerNorm` with learnable weight/bias | Pass weight, bias, normalized_shape, eps as parameters |

### 2. Embedding/Positional Encoding Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 5 | `embeddings/_1_RotaryEmbedding.py` | RotaryEmbedding | `register_buffer` for precomputed `inv_freq`, `cos_cached`, `sin_cached` | Pass cos/sin tensors as parameters, or compute on-the-fly |
| 6 | `embeddings/_3_SinusoidalPosEmbed.py` | SinusoidalPosEmbed | `register_buffer` for precomputed `pe` table | Pass precomputed positional embeddings as parameter |
| 7 | `embeddings/_4_RelativePositionBias.py` | RelativePositionBias | `nn.Parameter` for bias table + `register_buffer` for position index | Pass bias_table, position_index as parameters |

### 3. Attention Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 8 | `attention/_6_ALiBi.py` | ALiBi | `nn.Linear` for Q/K/V/O projections + `register_buffer` for slopes | Pass projection weights and slopes as parameters |
| 9 | `attention/_7_SlidingWindowAttention.py` | SlidingWindowAttention | `nn.Linear` for Q/K/V/O projections + stored config | Pass projection weights and config as parameters |

### 4. MoE Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 10 | `moe/_1_TopK_Router.py` | TopKRouter | `nn.Linear` for gating + stored `top_k` config | Pass gate weights and top_k as parameters |

### 5. Convolution Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 11 | `convolutions/_1_Conv2d_Standard.py` | Conv2d | `nn.Conv2d` with learnable weights | Pass weight, bias, stride, padding, etc. as parameters |

### 6. Pooling Operators

| # | File Path | Operator Name | Current State | Refactoring Needed |
|---|-----------|---------------|---------------|-------------------|
| 12 | `pooling/_2_MaxPool2d.py` | MaxPool2d | Wraps `nn.MaxPool2d` with stored kernel_size, stride, padding, dilation | Pass config parameters at call time |
| 13 | `pooling/_7_AdaptiveAvgPool2d.py` | AdaptiveAvgPool2d | Wraps `nn.AdaptiveAvgPool2d` with stored output_size | Pass output_size as parameter at call time |

---

## Refactoring Guidelines

When refactoring an operator to be functional:

1. **Remove stored config from `__init__`**: Move config parameters (like `eps`, `num_features`, `kernel_size`) to be passed as arguments to `forward()`.

2. **Remove learnable parameters**: Operators should not define `nn.Parameter` or use `register_buffer`. These should be managed by the level4 model and passed to the operator.

3. **Remove wrapped `nn.Module` subclasses**: Instead of wrapping `nn.BatchNorm2d`, implement the batch norm computation directly using the passed weight, bias, running stats.

4. **Functional signature pattern**:
   ```python
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
       
       def forward(self, x, weight=None, bias=None, **config):
           # Compute operation using x and passed parameters
           return result
   ```

5. **Update level4 models**: Level4 models will need to:
   - Store the learnable parameters (weight, bias, etc.)
   - Precompute any buffers (cos/sin for RoPE, slopes for ALiBi)
   - Pass these to the level1 operator at each forward call

---

## Notes

- The distinction between "functional" and "stateful" is important for kernel benchmarking: functional operators isolate the compute kernel from memory management of parameters.
- Some operators (like attention operators with Q/K/V projections) may need more significant restructuring since they currently bundle projection layers with the attention computation.
- Consider creating separate operators for projections vs. attention computation to achieve true functional decomposition.
