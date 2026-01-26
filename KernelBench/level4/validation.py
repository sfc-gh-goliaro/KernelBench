"""
Validation Utilities for Level4 Models

This module provides utilities for validating level4 model implementations
against reference implementations (HuggingFace, GitHub repos, etc.):
- Output comparison with tolerance
- Layer-by-layer activation comparison
- Performance metrics collection
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    max_diff: float
    mean_diff: float
    rtol: float
    atol: float
    num_elements: int
    details: Dict[str, Any]
    
    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"ValidationResult({status}, max_diff={self.max_diff:.2e}, "
            f"mean_diff={self.mean_diff:.2e}, elements={self.num_elements})"
        )


def compare_tensors(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationResult:
    """
    Compare two tensors with given tolerances.
    
    Args:
        output: Output tensor from model being tested
        reference: Reference tensor to compare against
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        ValidationResult with comparison details
    """
    # Ensure same device and dtype for comparison
    if output.device != reference.device:
        reference = reference.to(output.device)
    if output.dtype != reference.dtype:
        reference = reference.to(output.dtype)
    
    # Handle shape mismatch
    if output.shape != reference.shape:
        return ValidationResult(
            is_valid=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            rtol=rtol,
            atol=atol,
            num_elements=0,
            details={
                "error": "Shape mismatch",
                "output_shape": list(output.shape),
                "reference_shape": list(reference.shape),
            }
        )
    
    # Compute differences
    diff = torch.abs(output - reference)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check tolerance
    # PyTorch allclose: |a - b| <= atol + rtol * |b|
    tolerance = atol + rtol * torch.abs(reference)
    is_valid = (diff <= tolerance).all().item()
    
    # Compute additional stats
    num_failures = (diff > tolerance).sum().item()
    
    return ValidationResult(
        is_valid=is_valid,
        max_diff=max_diff,
        mean_diff=mean_diff,
        rtol=rtol,
        atol=atol,
        num_elements=output.numel(),
        details={
            "num_failures": num_failures,
            "failure_rate": num_failures / output.numel() if output.numel() > 0 else 0,
            "output_norm": output.norm().item(),
            "reference_norm": reference.norm().item(),
        }
    )


def compare_model_outputs(
    model: nn.Module,
    reference_model: nn.Module,
    inputs: Union[torch.Tensor, Tuple, List],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationResult:
    """
    Compare outputs of two models given the same inputs.
    
    Args:
        model: Model being tested
        reference_model: Reference model
        inputs: Input tensor(s) for both models
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        ValidationResult with comparison details
    """
    model.eval()
    reference_model.eval()
    
    with torch.no_grad():
        # Handle different input types
        if isinstance(inputs, (tuple, list)):
            output = model(*inputs)
            reference = reference_model(*inputs)
        else:
            output = model(inputs)
            reference = reference_model(inputs)
    
    return compare_tensors(output, reference, rtol, atol)


class LayerOutputCapture:
    """Context manager to capture intermediate layer outputs."""
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.outputs: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
    
    def __enter__(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(
                    lambda m, inp, out, name=name: self.outputs.update({name: out})
                )
                self.hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()


def compare_layer_by_layer(
    model: nn.Module,
    reference_model: nn.Module,
    inputs: Union[torch.Tensor, Tuple, List],
    layer_names: Optional[List[str]] = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, ValidationResult]:
    """
    Compare layer-by-layer outputs between two models.
    
    Args:
        model: Model being tested
        reference_model: Reference model
        inputs: Input tensor(s)
        layer_names: List of layer names to compare. If None, compares all layers.
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary mapping layer names to ValidationResults
    """
    model.eval()
    reference_model.eval()
    
    # Get all layer names if not specified
    if layer_names is None:
        layer_names = [name for name, _ in model.named_modules() if name]
    
    results = {}
    
    with LayerOutputCapture(model, layer_names) as model_capture:
        with LayerOutputCapture(reference_model, layer_names) as ref_capture:
            with torch.no_grad():
                if isinstance(inputs, (tuple, list)):
                    _ = model(*inputs)
                    _ = reference_model(*inputs)
                else:
                    _ = model(inputs)
                    _ = reference_model(inputs)
            
            # Compare captured outputs
            for name in layer_names:
                if name in model_capture.outputs and name in ref_capture.outputs:
                    result = compare_tensors(
                        model_capture.outputs[name],
                        ref_capture.outputs[name],
                        rtol, atol
                    )
                    results[name] = result
    
    return results


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    forward_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    num_parameters: int
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(time={self.forward_time_ms:.2f}ms, "
            f"mem={self.memory_allocated_mb:.1f}MB, params={self.num_parameters:,})"
        )


def benchmark_model(
    model: nn.Module,
    inputs: Union[torch.Tensor, Tuple, List],
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cuda",
) -> PerformanceMetrics:
    """
    Benchmark model performance.
    
    Args:
        model: Model to benchmark
        inputs: Input tensor(s)
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations
        device: Device to run on
        
    Returns:
        PerformanceMetrics with timing and memory info
    """
    model.eval()
    model = model.to(device)
    
    # Move inputs to device
    if isinstance(inputs, (tuple, list)):
        inputs = tuple(
            x.to(device) if isinstance(x, torch.Tensor) else x 
            for x in inputs
        )
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, (tuple, list)):
                _ = model(*inputs)
            else:
                _ = model(inputs)
    
    # Synchronize and clear cache
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            if isinstance(inputs, (tuple, list)):
                _ = model(*inputs)
            else:
                _ = model(inputs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times.append((time.perf_counter() - start) * 1000)
    
    # Get memory stats
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return PerformanceMetrics(
        forward_time_ms=sum(times) / len(times),
        memory_allocated_mb=memory_allocated,
        memory_reserved_mb=memory_reserved,
        num_parameters=sum(p.numel() for p in model.parameters()),
    )


def validate_against_huggingface(
    model: nn.Module,
    hf_model_name: str,
    inputs: Union[torch.Tensor, Tuple, List],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    load_weights: bool = False,
) -> ValidationResult:
    """
    Validate model against HuggingFace implementation.
    
    Args:
        model: Model being tested
        hf_model_name: HuggingFace model name/path
        inputs: Input tensor(s)
        rtol: Relative tolerance
        atol: Absolute tolerance
        load_weights: Whether to load pretrained weights
        
    Returns:
        ValidationResult with comparison details
    """
    try:
        from transformers import AutoModel, AutoModelForCausalLM
        
        # Try to load appropriate model type
        try:
            if load_weights:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name, trust_remote_code=True
                )
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
                hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        except:
            if load_weights:
                hf_model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
                hf_model = AutoModel.from_config(config, trust_remote_code=True)
        
        # If not loading weights, copy weights from our model to HF model
        # (for architecture comparison only)
        if not load_weights:
            # This would require weight mapping - for now, just compare random init
            pass
        
        return compare_model_outputs(model, hf_model, inputs, rtol, atol)
        
    except Exception as e:
        logger.error(f"Failed to validate against HuggingFace: {e}")
        return ValidationResult(
            is_valid=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            rtol=rtol,
            atol=atol,
            num_elements=0,
            details={"error": str(e)}
        )


# ============================================================================
# End-to-End Validation with Weight Transfer
# ============================================================================

def validate_with_hf_weights(
    model: nn.Module,
    hf_model_name: str,
    batch_size: int = 1,
    seq_len: int = 128,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    compare_logits: bool = True,
) -> ValidationResult:
    """
    Validate a level4 model against HuggingFace by loading the same weights.
    
    This is the main entry point for end-to-end correctness validation.
    It loads HuggingFace weights into both models and compares their outputs.
    
    Args:
        model: Level4 model instance (must implement get_weight_mapper())
        hf_model_name: HuggingFace model name/path
        batch_size: Batch size for test inputs
        seq_len: Sequence length for test inputs
        device: Device to run validation on
        dtype: Data type for computation
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        compare_logits: If True, compare logits; else compare hidden states
        
    Returns:
        ValidationResult with comparison details
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading HuggingFace model: {hf_model_name}")
        
        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        hf_model.eval()
        
        # Get HuggingFace state dict
        hf_state_dict = hf_model.state_dict()
        
        # Transfer weights to our model
        logger.info("Transferring weights to level4 model...")
        from .base import transfer_weights
        
        weight_mapper = model.get_weight_mapper()
        num_layers = getattr(model.config, 'num_layers', 32)
        
        missing, unexpected = transfer_weights(
            hf_state_dict,
            model,
            weight_mapper,
            num_layers,
            strict=False,
        )
        
        logger.info(f"Weight transfer complete. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Move our model to device
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # Create test inputs
        vocab_size = getattr(model.config, 'vocab_size', 32000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Run forward pass on both models
        with torch.no_grad():
            # Our model output
            our_output = model(input_ids)
            
            # HuggingFace model output
            hf_output = hf_model(input_ids)
            
            if compare_logits:
                hf_logits = hf_output.logits
            else:
                # Get last hidden state if available
                hf_logits = hf_output.hidden_states[-1] if hasattr(hf_output, 'hidden_states') else hf_output.logits
        
        # Compare outputs
        result = compare_tensors(our_output, hf_logits, rtol=rtol, atol=atol)
        
        # Add additional details
        result.details.update({
            "missing_weights": len(missing),
            "unexpected_weights": len(unexpected),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": str(device),
            "dtype": str(dtype),
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return ValidationResult(
            is_valid=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            rtol=rtol,
            atol=atol,
            num_elements=0,
            details={"error": str(e)}
        )


def validate_level4_model(
    model_class,
    variant: str,
    batch_size: int = 1,
    seq_len: int = 64,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> ValidationResult:
    """
    Convenience function to validate a level4 model class against HuggingFace.
    
    Args:
        model_class: The level4 model class (e.g., from 1_Llama31 import Model)
        variant: Variant name (e.g., "8B")
        batch_size: Batch size for test inputs
        seq_len: Sequence length for test inputs
        device: Device to run validation on
        dtype: Data type for computation
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        ValidationResult with comparison details
        
    Example:
        >>> from KernelBench.level4 import _1_Llama31
        >>> result = validate_level4_model(_1_Llama31.Model, "8B")
        >>> print(result)
    """
    # Create model from variant
    model = model_class.from_pretrained(variant)
    
    # Get HuggingFace model name
    hf_model_name = model_class.get_hf_model_name(variant)
    
    if not hf_model_name:
        return ValidationResult(
            is_valid=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            rtol=rtol,
            atol=atol,
            num_elements=0,
            details={"error": f"No HuggingFace model name found for variant: {variant}"}
        )
    
    return validate_with_hf_weights(
        model=model,
        hf_model_name=hf_model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )


def run_full_validation_suite(
    models: Optional[List[Tuple[type, str]]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Dict[str, ValidationResult]:
    """
    Run validation suite on multiple level4 models.
    
    Args:
        models: List of (model_class, variant) tuples. If None, runs on all available.
        device: Device to run validation on
        dtype: Data type for computation
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary mapping model names to ValidationResults
    """
    if models is None:
        # Import all level4 models
        from . import (
            _1_Llama31, _2_Falcon, _3_Mistral, _4_MoE,
            _9_T5, _12_Whisper,
        )
        models = [
            (_1_Llama31.Model, "8B"),
            (_2_Falcon.Model, "7B"),
            (_3_Mistral.Model, "7B"),
        ]
    
    results = {}
    
    for model_class, variant in models:
        model_name = f"{model_class.__module__.split('.')[-1]}_{variant}"
        logger.info(f"Validating {model_name}...")
        
        try:
            result = validate_level4_model(
                model_class=model_class,
                variant=variant,
                device=device,
                dtype=dtype,
                rtol=rtol,
                atol=atol,
            )
            results[model_name] = result
            logger.info(f"{model_name}: {result}")
        except Exception as e:
            logger.error(f"{model_name}: Failed - {e}")
            results[model_name] = ValidationResult(
                is_valid=False,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                rtol=rtol,
                atol=atol,
                num_elements=0,
                details={"error": str(e)}
            )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results.values() if r.is_valid)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    for name, result in results.items():
        status = "PASS" if result.is_valid else "FAIL"
        print(f"  {name}: {status} (max_diff={result.max_diff:.2e})")
    print("=" * 60)
    
    return results


def run_validation_suite(
    model: nn.Module,
    test_cases: List[Dict[str, Any]],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, ValidationResult]:
    """
    Run a suite of validation tests on a model.
    
    Args:
        model: Model to validate
        test_cases: List of test case dictionaries with 'name', 'inputs', 'expected' keys
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary mapping test names to ValidationResults
    """
    results = {}
    
    model.eval()
    for test_case in test_cases:
        name = test_case.get("name", f"test_{len(results)}")
        inputs = test_case["inputs"]
        expected = test_case.get("expected")
        
        with torch.no_grad():
            if isinstance(inputs, (tuple, list)):
                output = model(*inputs)
            else:
                output = model(inputs)
        
        if expected is not None:
            result = compare_tensors(output, expected, rtol, atol)
        else:
            # Just check that output is valid (no NaN/Inf)
            is_valid = not (torch.isnan(output).any() or torch.isinf(output).any())
            result = ValidationResult(
                is_valid=is_valid,
                max_diff=0.0,
                mean_diff=0.0,
                rtol=rtol,
                atol=atol,
                num_elements=output.numel(),
                details={"has_nan": torch.isnan(output).any().item()}
            )
        
        results[name] = result
    
    return results
