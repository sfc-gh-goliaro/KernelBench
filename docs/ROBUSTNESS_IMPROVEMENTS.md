# KernelBench v2: Robustness Improvements

This document describes the changes made to KernelBench to improve evaluation robustness, inspired by the analysis in [robust-kbench](https://github.com/SqueezeAILab/robust-kbench).

## Overview

KernelBench v2 introduces dimension-agnostic comparison modes, anti-exploit filters, and task validation infrastructure to ensure robust evaluation of LLM-generated CUDA kernels.

## Feature Comparison

| Feature | KernelBench v1 | robust-kbench | KernelBench v2 |
|---------|----------------|---------------|----------------|
| **Correctness Checking** |
| Standard `torch.allclose` | ✅ | ✅ | ✅ |
| Log-domain comparison | ❌ | ❌ | ✅ |
| Relative-only comparison | ❌ | ❌ | ✅ |
| Top-k ranking comparison | ❌ | ❌ | ✅ |
| Distribution comparison | ❌ | ❌ | ✅ |
| Per-task comparison mode config | ❌ | ❌ | ✅ |
| **Task Validation Filters** |
| Output range check | ❌ | ✅ | ✅ |
| Output std check | ❌ | ✅ | ✅ |
| Output axes variation | ❌ | ✅ | ✅ |
| Input impact check | ❌ | ✅ | ✅ |
| LLM sanity check | ❌ | ✅ | ❌ (not needed) |
| **Anti-Exploit Checks** |
| Always-zero detection | ❌ | ❌ | ✅ |
| Constant output detection | ❌ | ❌ | ✅ |
| Ignores-input detection | ❌ | ❌ | ✅ |
| Ignores-weights detection | ❌ | ❌ | ✅ |
| **Infrastructure** |
| Configurable input shapes | ❌ | ✅ | ✅ |
| Task-level config (TASK_CONFIG) | ❌ | ✅ (JSON files) | ✅ (in .py files) |
| Validation script | ❌ | ✅ | ✅ |
| **Task Coverage** |
| Fixes for problematic tasks | ❌ | Filter out | ✅ Fix in-place |
| All 394 tasks usable | ✅ | ❌ (~50 filtered) | ✅ |

## New Files

### `src/filters.py`

Core filtering and comparison infrastructure:

```
src/filters.py
├── Comparison Modes
│   ├── compare_default()        # Standard torch.allclose
│   ├── compare_log_domain()     # Log-space comparison for softmax
│   ├── compare_relative_only()  # Pure relative tolerance
│   ├── compare_topk_ranking()   # Top-k values and rankings
│   └── compare_distribution()   # Statistical properties
├── Task Validation Filters
│   ├── collect_outputs()        # Single-pass output collection
│   ├── analyze_output_range()   # Check if outputs in (-0.01, 0.01)
│   ├── analyze_output_std()     # Check variance across seeds
│   ├── analyze_output_axes()    # Check variance across axes
│   ├── analyze_input_impact()   # Check if inputs affect output
│   ├── run_all_filters()        # Efficient combined analysis
│   └── validate_task()          # Complete task validation
└── Anti-Exploit Checks
    ├── check_always_zero()      # Detect zero-output exploits
    ├── check_constant_output()  # Detect constant outputs
    └── run_anti_exploit_checks() # All exploit checks
```

### `scripts/validate_tasks.py`

Task validation script with multi-GPU support:

```bash
# Validate specific levels
python scripts/validate_tasks.py --level 1

# Validate all levels with 4 GPUs
python scripts/validate_tasks.py --all --num-gpus 4

# Verbose output
python scripts/validate_tasks.py --all -v

# Save results to JSON
python scripts/validate_tasks.py --all --output results.json
```

## Comparison Modes

### Problem: Numerical Instability

Standard `torch.allclose(ref, test, atol=1e-4, rtol=1e-4)` fails for:

1. **Softmax outputs**: With 16K classes, most probabilities are ~6e-5. A small error (1e-5) is 16% relative error.
2. **Normalized outputs**: L1/L2/Frobenius norms produce intentionally small values.
3. **Product reductions**: `cumprod` can produce extremely small or large values.

### Solution: Task-Specific Comparison Modes

Each task can specify a `comparison_mode` in its `TASK_CONFIG`:

```python
# In task file
TASK_CONFIG = {
    'comparison_mode': 'log_domain',  # or 'relative', 'topk', 'distribution'
    'atol': 1e-4,
    'rtol': 1e-4,
}
```

| Mode | Use Case | How It Works |
|------|----------|--------------|
| `default` | Most tasks | Standard `torch.allclose` |
| `log_domain` | Softmax, products | Compare `log(output)` instead of `output` |
| `relative` | Normalized outputs | Pure relative error, no absolute tolerance |
| `topk` | Classification | Compare only top-k values and rankings |
| `distribution` | Probability distributions | Compare entropy, argmax, moments |

### Example: Softmax

```python
# Before (fails with large dimensions)
torch.allclose(ref, test, atol=1e-4, rtol=1e-4)  # ❌ Fails

# After (log-domain comparison)
log_ref = torch.log(ref + eps)
log_test = torch.log(test + eps)
torch.allclose(log_ref, log_test, atol=1e-4, rtol=1e-4)  # ✅ Passes
```

## Task Validation Filters

### Filter 1: Output Range

Detects if all outputs are in `(-0.01, 0.01)`, which could indicate:
- Task produces near-zero outputs (exploitable)
- LLM could cheat by always outputting zeros

```python
def analyze_output_range(all_outputs):
    min_val = all_outputs.min().item()
    max_val = all_outputs.max().item()
    return min_val > -0.01 and max_val < 0.01
```

### Filter 2: Output Standard Deviation

Detects if outputs don't vary across different random seeds:

```python
def analyze_output_std(all_outputs):
    var = torch.var(flat, dim=0)
    max_var = var.max().item()
    return max_var < 0.0001  # std < 0.01
```

### Filter 3: Output Axes Variation

Detects if outputs are constant along any axis:

```python
def analyze_output_axes(all_outputs):
    var = torch.var(flat, dim=0)
    max_var = var.max().item()
    return max_var < 0.001  # Very low variation
```

### Filter 4: Input Impact

Detects if different inputs produce the same output (model ignores inputs):

```python
def analyze_input_impact(all_outputs):
    # Same as output_std - if outputs are identical across
    # different random inputs, the model ignores them
    return analyze_output_std(all_outputs)
```

## Tasks Updated with TASK_CONFIG

All 394 tasks in levels 1-7 have been updated with:

1. `TASK_CONFIG` dictionary with appropriate `comparison_mode`
2. `get_inputs(**kwargs)` signature for configurable dimensions

### Problematic Tasks Fixed

| Task | Problem | Fix |
|------|---------|-----|
| `23_Softmax.py` | Small probabilities | `comparison_mode: 'log_domain'` |
| `24_LogSoftmax.py` | Already log-space | `comparison_mode: 'log_domain'` |
| `37_FrobeniusNorm_.py` | Small normalized values | `comparison_mode: 'relative'` |
| `38_L1Norm_.py` | Small normalized values | `comparison_mode: 'relative'` |
| `40_LayerNorm.py` | Normalized outputs | `comparison_mode: 'relative'` |
| `41_BatchNorm.py` | Normalized outputs | `comparison_mode: 'relative'` |
| `42_GroupNorm.py` | Normalized outputs | `comparison_mode: 'relative'` |
| `43_InstanceNorm.py` | Normalized outputs | `comparison_mode: 'relative'` |
| `44_LocalResponseNorm.py` | Normalized outputs | `comparison_mode: 'relative'` |
| `89_cumsum.py` | Accumulated values | `comparison_mode: 'relative'` |
| `90_cumprod.py` | Very small/large products | `comparison_mode: 'log_domain'` |
| `95_CrossEntropyLoss.py` | Loss values | `comparison_mode: 'relative'` |
| `96_HuberLoss.py` | Loss values | `comparison_mode: 'relative'` |
| `97_CosineSimilarityLoss.py` | Similarity values | `comparison_mode: 'relative'` |
| `98_KLDivLoss.py` | Divergence values | `comparison_mode: 'relative'` |

### Level 2+ Combined Operations

Many Level 2+ tasks combine softmax/normalization with other ops:

| Pattern | comparison_mode |
|---------|-----------------|
| `*_Softmax_*` | `log_domain` |
| `*_BatchNorm_*`, `*_LayerNorm_*` | `relative` |
| `*_RMSNorm_*` | `relative` |
| `*_Dropout_*_Softmax_*` | `topk` |
| `*_Softmax_*_Sigmoid_*` | `distribution` |

## Anti-Exploit Checks

These checks detect LLM-generated kernels that "cheat" by exploiting evaluation weaknesses:

### Check 1: Always Zero

Detects if output is mostly zeros:

```python
def check_always_zero(output, threshold=1e-6):
    zero_fraction = (torch.abs(output) < threshold).float().mean()
    return zero_fraction > 0.99
```

### Check 2: Constant Output

Detects if model produces same output regardless of input:

```python
def check_constant_output(outputs, threshold=1e-6):
    stacked = torch.stack([o.flatten() for o in outputs])
    std = torch.std(stacked, dim=0).mean()
    return std < threshold
```

### Check 3: Ignores Weights

Detects if model output is independent of weight initialization:

```python
def check_ignores_weights(model_cls, init_inputs, inputs, num_seeds=3):
    outputs = []
    for seed in range(num_seeds):
        set_seed(seed)
        model = model_cls(*init_inputs).cuda().eval()
        outputs.append(model(*inputs))
    return check_constant_output(outputs)
```

## Usage

### Validate Tasks

```bash
# Quick validation of level 1
python scripts/validate_tasks.py --level 1

# Full validation with all GPUs
python scripts/validate_tasks.py --all --num-gpus 8 -v

# Save results
python scripts/validate_tasks.py --all --output validation_results.json
```

### Use Comparison Modes in Evaluation

```python
from src.filters import get_comparison_fn, COMPARISON_MODES

# Get comparison function for a task
task_config = {'comparison_mode': 'log_domain', 'atol': 1e-4, 'rtol': 1e-4}
compare_fn = get_comparison_fn(task_config['comparison_mode'])

# Use in correctness check
is_correct = compare_fn(ref_output, test_output, **task_config)
```

### Run Anti-Exploit Checks

```python
from src.filters import run_anti_exploit_checks

results = run_anti_exploit_checks(
    model=my_model,
    get_inputs=get_inputs_fn,
    model_cls=Model,
    init_inputs=init_args,
    device='cuda'
)

if results['exploits_detected']:
    print("Exploit detected:", results['exploit_warnings'])
```

## Comparison with robust-kbench

| Aspect | robust-kbench | KernelBench v2 |
|--------|---------------|----------------|
| Approach to problematic tasks | Filter out (~50 tasks) | Fix with comparison modes (0 filtered) |
| Task configuration | Separate JSON files | Embedded in task .py files |
| LLM sanity check | ✅ (requires API) | ❌ (not needed - fixes at source) |
| New comparison modes | ❌ | ✅ (5 modes) |
| Anti-exploit checks | Implicit via filters | ✅ Explicit checks |

## Conclusion

KernelBench v2 achieves the same robustness goals as robust-kbench but with a different philosophy:

- **robust-kbench**: Filter out problematic tasks
- **KernelBench v2**: Fix problematic tasks with appropriate comparison modes

This approach keeps all 394 tasks usable while ensuring robust evaluation.

