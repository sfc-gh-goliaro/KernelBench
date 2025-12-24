"""
Robustness filters and comparison modes for KernelBench.

This module provides:
1. Dimension-agnostic comparison modes for correctness checking
2. Task validation filters (like robust-kbench)
3. Anti-exploit checks for LLM-generated kernels
"""

import torch
import torch.nn.functional as F
import importlib.util
import os
from typing import Dict, List, Callable, Optional, Any, Tuple


# =============================================================================
# COMPARISON MODES - Dimension-agnostic correctness checking
# =============================================================================

def compare_default(ref: torch.Tensor, test: torch.Tensor, 
                    atol: float = 1e-4, rtol: float = 1e-4, **kwargs) -> bool:
    """Standard torch.allclose comparison."""
    return torch.allclose(ref, test, atol=atol, rtol=rtol)


def compare_log_domain(ref: torch.Tensor, test: torch.Tensor,
                       atol: float = 1e-4, rtol: float = 1e-4, **kwargs) -> bool:
    """
    Compare tensors in log-domain for numerical stability.
    
    Useful for softmax outputs where values can be extremely small (e.g., 1/16384).
    In log-space, these become reasonable numbers like -9.7.
    """
    eps = 1e-10
    # Handle negative values by comparing absolute values in log space
    ref_safe = torch.abs(ref) + eps
    test_safe = torch.abs(test) + eps
    
    log_ref = torch.log(ref_safe)
    log_test = torch.log(test_safe)
    
    # Also check signs match for non-tiny values
    sign_mask = (torch.abs(ref) > eps) & (torch.abs(test) > eps)
    if sign_mask.any():
        signs_match = (torch.sign(ref[sign_mask]) == torch.sign(test[sign_mask])).all()
        if not signs_match:
            return False
    
    return torch.allclose(log_ref, log_test, atol=atol, rtol=rtol)


def compare_relative_only(ref: torch.Tensor, test: torch.Tensor,
                          rtol: float = 1e-3, atol: float = 1e-6, **kwargs) -> bool:
    """
    Pure relative comparison - useful for normalized outputs.
    
    For outputs that are intentionally small (e.g., after normalization),
    we care about relative error, not absolute differences.
    """
    min_val = kwargs.get('min_val', 1e-10)
    
    ref_abs = torch.abs(ref)
    diff = torch.abs(ref - test)
    
    # Relative error with floor to avoid division by zero
    rel_error = diff / torch.clamp(ref_abs, min=min_val)
    
    # For very small values, fall back to absolute comparison
    small_mask = ref_abs < min_val
    if small_mask.any():
        if not (diff[small_mask] < atol).all():
            return False
    
    # Check relative error for non-small values
    large_mask = ~small_mask
    if large_mask.any():
        if not (rel_error[large_mask] < rtol).all():
            return False
    
    return True


def compare_topk_ranking(ref: torch.Tensor, test: torch.Tensor,
                         atol: float = 1e-4, rtol: float = 1e-3, **kwargs) -> bool:
    """
    Compare only top-k values and rankings.
    
    For softmax outputs in classification, only the top-k probabilities matter.
    The remaining near-zero values are numerically unstable anyway.
    """
    k = kwargs.get('topk_k', 10)
    
    # Handle different tensor shapes
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
        test = test.unsqueeze(0)
    
    # Use last dimension for topk
    actual_k = min(k, ref.size(-1))
    
    # Get top-k from reference
    ref_vals, ref_idx = torch.topk(ref, actual_k, dim=-1)
    
    # Get top-k from test
    test_vals, test_idx = torch.topk(test, actual_k, dim=-1)
    
    # Check ranking matches (at least 90%)
    ranking_match = (ref_idx == test_idx).float().mean().item()
    if ranking_match < 0.9:
        return False
    
    # Check top-k values match relatively
    test_at_ref_idx = torch.gather(test, -1, ref_idx)
    if not torch.allclose(ref_vals, test_at_ref_idx, atol=atol, rtol=rtol):
        return False
    
    return True


def compare_distribution(ref: torch.Tensor, test: torch.Tensor,
                         atol: float = 1e-4, rtol: float = 1e-3, **kwargs) -> bool:
    """
    Compare statistical properties of distributions.
    
    For probability distributions, compare entropy, max, argmax rather than
    exact element-wise values.
    """
    eps = 1e-10
    
    # Flatten to 2D for consistent handling
    orig_shape = ref.shape
    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
        test = test.unsqueeze(0)
    elif ref.dim() > 2:
        ref = ref.view(-1, ref.size(-1))
        test = test.view(-1, test.size(-1))
    
    # Entropy comparison
    ref_entropy = -(ref * torch.log(ref + eps)).sum(dim=-1)
    test_entropy = -(test * torch.log(test + eps)).sum(dim=-1)
    if not torch.allclose(ref_entropy, test_entropy, atol=atol, rtol=rtol):
        return False
    
    # Max value comparison
    ref_max = ref.max(dim=-1).values
    test_max = test.max(dim=-1).values
    if not torch.allclose(ref_max, test_max, atol=atol, rtol=rtol):
        return False
    
    # Argmax comparison (classification decision)
    ref_argmax = ref.argmax(dim=-1)
    test_argmax = test.argmax(dim=-1)
    if not (ref_argmax == test_argmax).all():
        return False
    
    return True


# Registry of comparison modes
COMPARISON_MODES: Dict[str, Callable] = {
    'default': compare_default,
    'log_domain': compare_log_domain,
    'relative': compare_relative_only,
    'topk': compare_topk_ranking,
    'distribution': compare_distribution,
}


def get_comparison_fn(mode: str) -> Callable:
    """Get comparison function by name, defaulting to standard comparison."""
    return COMPARISON_MODES.get(mode, compare_default)


# =============================================================================
# TASK VALIDATION FILTERS - Identify problematic tasks
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def collect_outputs(model: torch.nn.Module, get_inputs: Callable,
                    num_seeds: int = 3, device: str = 'cuda') -> torch.Tensor:
    """
    Collect model outputs for multiple random seeds.
    
    This is the single point where model forward passes are executed.
    All filter checks should use outputs from this function to avoid redundant computation.
    
    IMPORTANT: Outputs are kept on GPU for fast analysis (variance, min, max).
    Only scalar results are moved to CPU via .item().
    
    Args:
        model: The model to run
        get_inputs: Function to generate inputs
        num_seeds: Number of different random seeds to use
        device: Device to run on
        
    Returns:
        Stacked tensor of shape (num_seeds, *output_shape) ON GPU
    """
    outputs = []
    
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            # Keep on GPU for fast analysis - don't call .cpu()!
            out = model(*inputs).float()
            outputs.append(out)
    
    return torch.stack(outputs)


# =============================================================================
# FILTER ANALYSIS FUNCTIONS - Operate on pre-collected outputs (no model runs)
# =============================================================================

# Thresholds
_STD_THRESHOLD = 0.01
_VAR_THRESHOLD = _STD_THRESHOLD ** 2  # 0.0001


def analyze_output_range(all_outputs: torch.Tensor) -> bool:
    """
    Check if all output values are within (-0.01, 0.01).
    
    Uses min/max reductions which are O(n) but very fast.
    
    Args:
        all_outputs: Tensor of shape (num_seeds, *output_shape)
        
    Returns:
        True if task should be filtered (problematic).
    """
    # Use min/max - much faster than element-wise comparison + all()
    min_val = all_outputs.min().item()
    max_val = all_outputs.max().item()
    return min_val > -0.01 and max_val < 0.01


def analyze_output_std(all_outputs: torch.Tensor) -> bool:
    """
    Check if outputs don't vary enough across different seeds.
    
    Uses variance with max reduction - if max(variance) < threshold,
    then all variances are below threshold.
    
    Args:
        all_outputs: Tensor of shape (num_seeds, *output_shape)
        
    Returns:
        True if task should be filtered (problematic).
    """
    num_seeds = all_outputs.shape[0]
    flat = all_outputs.view(num_seeds, -1)
    
    # Compute variance across seeds, then take max
    var = torch.var(flat, dim=0)
    max_var = var.max().item()
    
    return max_var < _VAR_THRESHOLD


def analyze_output_axes(all_outputs: torch.Tensor) -> bool:
    """
    Check if outputs don't vary enough across different axes.
    
    Args:
        all_outputs: Tensor of shape (num_seeds, *output_shape)
        
    Returns:
        True if task should be filtered (problematic).
    """
    num_seeds = all_outputs.shape[0]
    flat = all_outputs.view(num_seeds, -1)
    
    # Compute variance across seeds
    var = torch.var(flat, dim=0)
    max_var = var.max().item()
    
    # If max variance is very low, it's problematic
    if max_var < _VAR_THRESHOLD:
        return True
    
    # Also flag suspiciously low variation (10x threshold)
    if max_var < _VAR_THRESHOLD * 10:
        return True
    
    return False


def analyze_input_impact(all_outputs: torch.Tensor) -> bool:
    """
    Check if inputs don't affect the output.
    
    This is the same as output_std - if outputs are the same across
    different random inputs (seeds), inputs don't impact the output.
    
    Args:
        all_outputs: Tensor of shape (num_seeds, *output_shape)
        
    Returns:
        True if task should be filtered (problematic).
    """
    return analyze_output_std(all_outputs)


def run_all_filters(all_outputs: torch.Tensor) -> Dict[str, bool]:
    """
    Run all filter analyses on pre-collected outputs efficiently.
    
    Optimizations:
    1. Computes variance ONCE, reused for output_std, input_impact, output_axes
    2. Uses max() reduction instead of .all() for speed
    3. Global min/max for range check (no intermediate boolean tensor)
    
    Args:
        all_outputs: Tensor of shape (num_seeds, *output_shape)
        
    Returns:
        Dict mapping filter names to results (True = problematic)
    """
    num_seeds = all_outputs.shape[0]
    
    # === Output Range Check ===
    # Global min/max - highly optimized single-pass CUDA kernels
    global_min = all_outputs.min().item()
    global_max = all_outputs.max().item()
    in_range = (global_min > -0.01) and (global_max < 0.01)
    
    # === Variance Check (shared across std/axes/input_impact) ===
    flat = all_outputs.view(num_seeds, -1)
    # Compute variance ONCE across seeds for each position
    var = torch.var(flat, dim=0)
    
    # Use max reduction - if max(var) < threshold, all vars are below threshold
    max_var = var.max().item()
    low_var = max_var < _VAR_THRESHOLD
    very_low_var = max_var < _VAR_THRESHOLD * 10
    
    return {
        'output_range': in_range,
        'output_std': low_var,
        'output_axes': low_var or very_low_var,
        'input_impact': low_var,
    }


def validate_task(task_path: str, device: str = 'cuda', num_seeds: int = 3) -> Dict[str, Any]:
    """
    Run all validation filters on a task efficiently.
    
    Collects outputs ONCE and runs all filter analyses on the same data.
    This is 4x faster than running each filter separately.
    
    Args:
        task_path: Path to the task Python file
        device: Device to run on ('cuda' or 'cpu')
        num_seeds: Number of random seeds to use
        
    Returns:
        Dict with filter results and recommendations
    """
    import gc
    
    model = None
    all_outputs = None
    
    # Load task module
    spec = importlib.util.spec_from_file_location("task_module", task_path)
    task_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_module)
    
    Model = task_module.Model
    get_inputs = task_module.get_inputs
    get_init_inputs = task_module.get_init_inputs
    
    # Get task config if present
    task_config = getattr(task_module, 'TASK_CONFIG', {})
    
    # Initialize model
    set_seed(42)
    init_inputs = get_init_inputs()
    model = Model(*init_inputs)
    model = model.to(device)
    model = model.eval()
    
    results = {
        'task_path': task_path,
        'task_name': os.path.basename(task_path),
        'has_task_config': bool(task_config),
        'filters': {},
        'recommendations': [],
    }
    
    try:
        # Collect outputs ONCE - this is where all model forward passes happen
        all_outputs = collect_outputs(model, get_inputs, num_seeds, device)
        
        # Run all filter analyses on the collected outputs (no model runs here)
        results['filters'] = run_all_filters(all_outputs)
        
        # Generate recommendations
        if results['filters']['output_range']:
            results['recommendations'].append("Use 'log_domain' or 'relative' comparison_mode")
        if results['filters']['output_std']:
            results['recommendations'].append("Output has low variance - consider 'relative' comparison")
        if results['filters']['output_axes']:
            results['recommendations'].append("Low variance along axes - consider 'distribution' comparison")
        if results['filters']['input_impact']:
            results['recommendations'].append("Inputs don't affect output - task may need restructuring")
        
        results['is_problematic'] = any(results['filters'].values())
        
    except Exception as e:
        results['error'] = str(e)
        results['is_problematic'] = True
    
    finally:
        # Clean up GPU memory to prevent OOM across many tasks
        if all_outputs is not None:
            del all_outputs
        if model is not None:
            del model
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# =============================================================================
# ANTI-EXPLOIT CHECKS - Detect LLM gaming behaviors at eval time
# =============================================================================

def check_always_zero(output: torch.Tensor, threshold: float = 1e-6) -> bool:
    """
    Check if output is always near zero (potential exploit).
    
    Returns True if exploit detected.
    """
    zero_fraction = (torch.abs(output) < threshold).float().mean().item()
    return zero_fraction > 0.99


def check_constant_output(outputs: List[torch.Tensor], threshold: float = 1e-6) -> bool:
    """
    Check if model outputs are constant across different inputs.
    
    Returns True if exploit detected.
    """
    if len(outputs) < 2:
        return False
    
    stacked = torch.stack([o.flatten() for o in outputs])
    std = torch.std(stacked, dim=0).mean().item()
    return std < threshold


def run_anti_exploit_checks(model: torch.nn.Module, get_inputs: Callable,
                            model_cls=None, init_inputs: List = None,
                            num_trials: int = 3, device: str = 'cuda') -> Dict[str, Any]:
    """
    Run all anti-exploit checks on a model efficiently.
    
    Minimizes model runs by:
    1. Collecting outputs once for input variation check
    2. Reusing outputs where possible
    
    Args:
        model: The model to check
        get_inputs: Function to generate inputs
        model_cls: Optional model class for weight independence check
        init_inputs: Optional init inputs for weight independence check
        num_trials: Number of trials for each check
        device: Device to run on
        
    Returns:
        Dict with check results and overall exploit detection flag
    """
    import gc
    
    outputs = []
    weight_outputs = []
    temp_model = None
    
    results = {
        'checks': {},
        'exploits_detected': False,
        'exploit_warnings': [],
    }
    
    try:
        # Collect outputs for multiple random inputs - single batch of model runs
        # Keep on GPU for fast analysis
        for seed in range(num_trials):
            set_seed(seed * 1000)
            inputs = get_inputs()
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            with torch.no_grad():
                out = model(*inputs)
                outputs.append(out)
        
        # Check 1: Always zero (use first output)
        always_zero = check_always_zero(outputs[0])
        results['checks']['always_zero'] = always_zero
        if always_zero:
            results['exploit_warnings'].append("Output is always near zero")
        
        # Check 2 & 4: Ignores input / Constant output (same check, reuse outputs)
        constant_output = check_constant_output(outputs)
        results['checks']['ignores_input'] = constant_output
        results['checks']['constant_output'] = constant_output
        if constant_output:
            results['exploit_warnings'].append("Output ignores input variations")
        
        # Check 3: Ignores weights (requires creating new model instances)
        if model_cls is not None and init_inputs is not None:
            # Need fresh inputs for fair comparison
            set_seed(42)
            fixed_inputs = get_inputs()
            fixed_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in fixed_inputs]
            
            for seed in range(num_trials):
                set_seed(seed * 1000)
                temp_model = model_cls(*init_inputs).to(device).eval()
                
                with torch.no_grad():
                    out = temp_model(*fixed_inputs)
                    weight_outputs.append(out)
                
                # Clean up temp model immediately
                del temp_model
                temp_model = None
            
            ignores_weights = check_constant_output(weight_outputs)
            results['checks']['ignores_weights'] = ignores_weights
            if ignores_weights:
                results['exploit_warnings'].append("Output ignores weight variations")
        
        results['exploits_detected'] = any(results['checks'].values())
        
    except Exception as e:
        results['error'] = str(e)
    
    finally:
        # Clean up GPU memory
        for out in outputs:
            del out
        outputs.clear()
        
        for out in weight_outputs:
            del out
        weight_outputs.clear()
        
        if temp_model is not None:
            del temp_model
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_recommended_comparison_mode(task_name: str) -> str:
    """
    Get recommended comparison mode based on task name heuristics.
    """
    task_lower = task_name.lower()
    
    # Softmax-related tasks
    if 'softmax' in task_lower and 'log' not in task_lower:
        return 'log_domain'
    
    # Normalization tasks
    if any(x in task_lower for x in ['norm', 'l1norm', 'l2norm', 'frobenius', 'rms']):
        return 'relative'
    
    # Product operations
    if 'prod' in task_lower or 'cumprod' in task_lower:
        return 'log_domain'
    
    # Loss functions that return scalars
    if any(x in task_lower for x in ['loss', 'kldiv', 'entropy']):
        return 'relative'
    
    # Dropout + softmax combinations
    if 'dropout' in task_lower and 'softmax' in task_lower:
        return 'topk'
    
    return 'default'
