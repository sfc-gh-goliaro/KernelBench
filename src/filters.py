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


def filter_output_range(model: torch.nn.Module, get_inputs: Callable,
                        num_seeds: int = 5, device: str = 'cuda') -> bool:
    """
    Filter tasks whose outputs are always within (-0.01, 0.01).
    
    Returns True if task should be filtered (problematic).
    """
    outputs = []
    
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs).float().cpu()
            outputs.append(out)
    
    all_outputs = torch.stack(outputs)
    # Check if ALL values are within (-0.01, 0.01)
    filter_catch = ((all_outputs > -0.01) & (all_outputs < 0.01)).all()
    
    return bool(filter_catch)


def filter_output_std(model: torch.nn.Module, get_inputs: Callable,
                      num_seeds: int = 5, device: str = 'cuda') -> bool:
    """
    Filter tasks whose outputs don't vary enough across different seeds.
    
    Returns True if task should be filtered (problematic).
    """
    outputs = []
    
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs).float().cpu()
            outputs.append(out)
    
    all_outputs = torch.stack(outputs)
    stds = torch.std(all_outputs, dim=0)
    filter_catch = (stds < 0.01).all()
    
    return bool(filter_catch)


def filter_output_axes(model: torch.nn.Module, get_inputs: Callable,
                       num_seeds: int = 5, device: str = 'cuda') -> bool:
    """
    Filter tasks whose outputs don't vary enough across different axes.
    
    Returns True if task should be filtered (problematic).
    """
    outputs = []
    
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs).float().cpu()
            outputs.append(out)
    
    all_outputs = torch.stack(outputs)
    
    # Calculate std across each axis
    for axis in range(all_outputs.ndim):
        axis_std = torch.std(all_outputs, dim=axis)
        if (axis_std < 0.01).all():
            return True
    
    return False


def filter_input_impact(model: torch.nn.Module, get_inputs: Callable,
                        num_seeds: int = 5, device: str = 'cuda') -> bool:
    """
    Filter tasks whose inputs don't affect the output.
    
    Returns True if task should be filtered (problematic).
    """
    outputs = []
    
    # Use same model, different inputs
    set_seed(42)
    
    for seed in range(num_seeds):
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs).float().cpu()
            outputs.append(out)
    
    all_outputs = torch.stack(outputs)
    stds = torch.std(all_outputs, dim=0)
    filter_catch = (stds < 0.01).all()
    
    return bool(filter_catch)


def validate_task(task_path: str, device: str = 'cuda') -> Dict[str, Any]:
    """
    Run all validation filters on a task.
    
    Returns dict with filter results and recommendations.
    """
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
    model = Model(*init_inputs).to(device).eval()
    
    results = {
        'task_path': task_path,
        'task_name': os.path.basename(task_path),
        'has_task_config': bool(task_config),
        'filters': {},
        'recommendations': [],
    }
    
    try:
        # Run filters
        results['filters']['output_range'] = filter_output_range(model, get_inputs, device=device)
        results['filters']['output_std'] = filter_output_std(model, get_inputs, device=device)
        results['filters']['output_axes'] = filter_output_axes(model, get_inputs, device=device)
        results['filters']['input_impact'] = filter_input_impact(model, get_inputs, device=device)
        
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


def check_ignores_input(model: torch.nn.Module, get_inputs: Callable,
                        num_trials: int = 3, device: str = 'cuda') -> bool:
    """
    Check if model ignores input variations (same output for different inputs).
    
    Returns True if exploit detected.
    """
    outputs = []
    
    for seed in range(num_trials):
        set_seed(seed * 1000)  # Use different seeds
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs).cpu()
            outputs.append(out)
    
    return check_constant_output(outputs)


def check_ignores_weights(model_cls, init_inputs: List, inputs: List,
                          num_seeds: int = 3, device: str = 'cuda') -> bool:
    """
    Check if model output is independent of weight initialization.
    
    Returns True if exploit detected.
    """
    outputs = []
    
    for seed in range(num_seeds):
        set_seed(seed * 1000)
        model = model_cls(*init_inputs).to(device).eval()
        
        inputs_device = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            out = model(*inputs_device).cpu()
            outputs.append(out)
    
    return check_constant_output(outputs)


def run_anti_exploit_checks(model: torch.nn.Module, get_inputs: Callable,
                            model_cls=None, init_inputs: List = None,
                            num_trials: int = 3, device: str = 'cuda') -> Dict[str, Any]:
    """
    Run all anti-exploit checks on a model.
    
    Returns dict with check results and overall exploit detection flag.
    """
    results = {
        'checks': {},
        'exploits_detected': False,
        'exploit_warnings': [],
    }
    
    try:
        # Get sample output
        set_seed(42)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        
        with torch.no_grad():
            sample_output = model(*inputs).cpu()
        
        # Check 1: Always zero
        always_zero = check_always_zero(sample_output)
        results['checks']['always_zero'] = always_zero
        if always_zero:
            results['exploit_warnings'].append("Output is always near zero")
        
        # Check 2: Ignores input
        ignores_input = check_ignores_input(model, get_inputs, num_trials, device)
        results['checks']['ignores_input'] = ignores_input
        if ignores_input:
            results['exploit_warnings'].append("Output ignores input variations")
        
        # Check 3: Ignores weights (if model_cls provided)
        if model_cls is not None and init_inputs is not None:
            ignores_weights = check_ignores_weights(model_cls, init_inputs, inputs, num_trials, device)
            results['checks']['ignores_weights'] = ignores_weights
            if ignores_weights:
                results['exploit_warnings'].append("Output ignores weight variations")
        
        # Check 4: Constant output
        outputs = []
        for seed in range(num_trials):
            set_seed(seed * 100)
            trial_inputs = get_inputs()
            trial_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in trial_inputs]
            with torch.no_grad():
                outputs.append(model(*trial_inputs).cpu())
        
        constant_output = check_constant_output(outputs)
        results['checks']['constant_output'] = constant_output
        if constant_output:
            results['exploit_warnings'].append("Output is constant across trials")
        
        results['exploits_detected'] = any(results['checks'].values())
        
    except Exception as e:
        results['error'] = str(e)
    
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

