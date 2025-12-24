#!/usr/bin/env python3
"""
Task validation script for KernelBench.

Runs robustness filters on all tasks to identify problematic ones and
suggests comparison modes to fix them.

Supports multi-GPU parallelization for faster validation.

Usage:
    python scripts/validate_tasks.py --level 1
    python scripts/validate_tasks.py --level 1 2 3 4 5 6 7
    python scripts/validate_tasks.py --all
    python scripts/validate_tasks.py --all --num-gpus 4
    python scripts/validate_tasks.py --task KernelBench/level1/23_Softmax.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def get_tasks_for_level(level: int) -> List[str]:
    """Get all task file paths for a given level."""
    level_dir = REPO_ROOT / "KernelBench" / f"level{level}"
    if not level_dir.exists():
        print(f"Warning: Level directory {level_dir} does not exist")
        return []
    
    tasks = sorted(level_dir.glob("*.py"))
    return [str(t) for t in tasks]


def validate_single_task(args: Tuple[str, str, int]) -> Tuple[str, Dict[str, Any]]:
    """
    Validate a single task. Designed to run in a separate process.
    
    Args:
        args: Tuple of (task_path, device, num_seeds)
        
    Returns:
        Tuple of (task_name, result_dict)
    """
    task_path, device, num_seeds = args
    task_name = os.path.basename(task_path)
    
    # Import here to avoid issues with multiprocessing
    from src.filters import validate_task, get_recommended_comparison_mode
    
    try:
        task_result = validate_task(task_path, device=device, num_seeds=num_seeds)
        
        # Add recommendation if problematic
        if task_result.get('is_problematic') and not task_result.get('error'):
            recommended = get_recommended_comparison_mode(task_name)
            task_result['recommendation'] = {
                'comparison_mode': recommended,
                'filters_failed': [k for k, v in task_result.get('filters', {}).items() if v],
                'suggestions': task_result.get('recommendations', []),
            }
        
        return (task_name, task_result)
        
    except Exception as e:
        return (task_name, {'error': str(e), 'is_problematic': True})


def validate_all_tasks_parallel(
    all_tasks: List[str],
    num_gpus: int = 1,
    num_seeds: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate all tasks using multiple GPUs in parallel.
    
    Args:
        all_tasks: List of task file paths
        num_gpus: Number of GPUs to use
        num_seeds: Number of random seeds per task
        verbose: Print progress
        
    Returns:
        Dict with validation results for each task
    """
    results = {
        'summary': {
            'total': len(all_tasks),
            'problematic': 0,
            'passed': 0,
            'errors': 0,
        },
        'tasks': {},
        'problematic_tasks': [],
        'recommendations': {},
    }
    
    if not all_tasks:
        return results
    
    # Prepare task arguments with round-robin GPU assignment
    task_args = []
    for i, task_path in enumerate(all_tasks):
        gpu_id = i % num_gpus
        device = f'cuda:{gpu_id}'
        task_args.append((task_path, device, num_seeds))
    
    # Use process pool for true parallelism (avoids GIL)
    # Each process gets its own GPU
    num_workers = min(num_gpus, len(all_tasks))
    
    from tqdm import tqdm
    
    if num_workers > 1:
        # Multi-GPU parallel execution
        # Use 'spawn' to avoid CUDA context issues
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(validate_single_task, args): args[0] 
                      for args in task_args}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Validating ({num_gpus} GPUs)"):
                task_name, task_result = future.result()
                _process_task_result(results, task_name, task_result, verbose)
    else:
        # Single GPU sequential execution
        for args in tqdm(task_args, desc="Validating"):
            task_name, task_result = validate_single_task(args)
            _process_task_result(results, task_name, task_result, verbose)
    
    return results


def _process_task_result(results: Dict, task_name: str, task_result: Dict, verbose: bool):
    """Process a single task result and update the results dict."""
    results['tasks'][task_name] = task_result
    
    if task_result.get('error'):
        results['summary']['errors'] += 1
        if verbose:
            print(f"  {task_name}: ERROR - {task_result['error']}")
    elif task_result.get('is_problematic'):
        results['summary']['problematic'] += 1
        results['problematic_tasks'].append(task_name)
        
        if 'recommendation' in task_result:
            results['recommendations'][task_name] = task_result['recommendation']
        
        if verbose:
            rec = task_result.get('recommendation', {})
            print(f"  {task_name}: PROBLEMATIC - recommend: {rec.get('comparison_mode', '?')}")
            for suggestion in rec.get('suggestions', []):
                print(f"    - {suggestion}")
    else:
        results['summary']['passed'] += 1


def validate_all_tasks(
    levels: List[int] = None,
    task_paths: List[str] = None,
    num_gpus: int = 1,
    num_seeds: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate all tasks and return results.
    
    Args:
        levels: List of level numbers to validate
        task_paths: Specific task paths to validate (overrides levels)
        num_gpus: Number of GPUs to use for parallel validation
        num_seeds: Number of random seeds per task
        verbose: Print progress
        
    Returns:
        Dict with validation results for each task
    """
    # Collect task paths
    all_tasks = []
    if task_paths:
        all_tasks = task_paths
    elif levels:
        for level in levels:
            all_tasks.extend(get_tasks_for_level(level))
    
    return validate_all_tasks_parallel(
        all_tasks=all_tasks,
        num_gpus=num_gpus,
        num_seeds=num_seeds,
        verbose=verbose,
    )


def print_summary(results: Dict[str, Any]):
    """Print validation summary."""
    summary = results['summary']
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks:      {summary['total']}")
    print(f"Passed:           {summary['passed']}")
    print(f"Problematic:      {summary['problematic']}")
    print(f"Errors:           {summary['errors']}")
    
    if results['problematic_tasks']:
        print("\n" + "-" * 60)
        print("PROBLEMATIC TASKS (need TASK_CONFIG with comparison_mode)")
        print("-" * 60)
        for task_name in results['problematic_tasks']:
            rec = results['recommendations'].get(task_name, {})
            mode = rec.get('comparison_mode', 'unknown')
            filters = rec.get('filters_failed', [])
            print(f"  {task_name}")
            print(f"    -> comparison_mode: '{mode}'")
            print(f"    -> failed filters: {filters}")
    
    print("=" * 60)


def generate_fix_snippet(task_name: str, recommendation: Dict) -> str:
    """Generate a code snippet to add TASK_CONFIG to a task file."""
    mode = recommendation.get('comparison_mode', 'default')
    
    snippet = f'''
# Add this near the top of {task_name}, after the imports:

TASK_CONFIG = {{
    'comparison_mode': '{mode}',
    'atol': 1e-4,
    'rtol': 1e-4,
}}
'''
    return snippet


def get_num_gpus() -> int:
    """Detect number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate KernelBench tasks for robustness issues"
    )
    parser.add_argument(
        '--level', '-l',
        type=int,
        nargs='+',
        help='Level(s) to validate (1-7)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Validate all levels (1-7)'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        nargs='+',
        help='Specific task file(s) to validate'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--num-gpus', '-g',
        type=int,
        default=None,
        help='Number of GPUs to use (default: auto-detect)'
    )
    parser.add_argument(
        '--num-seeds', '-n',
        type=int,
        default=3,
        help='Number of random seeds per task (default: 3)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--show-fixes',
        action='store_true',
        help='Show code snippets to fix problematic tasks'
    )
    
    args = parser.parse_args()
    
    # Determine which tasks to validate
    levels = None
    task_paths = None
    
    if args.task:
        task_paths = args.task
    elif args.all:
        levels = list(range(1, 8))
    elif args.level:
        levels = args.level
    else:
        print("Error: Must specify --level, --all, or --task")
        parser.print_help()
        sys.exit(1)
    
    # Detect or use specified number of GPUs
    num_gpus = args.num_gpus if args.num_gpus else get_num_gpus()
    num_gpus = max(1, num_gpus)  # At least 1
    
    # Run validation
    print("Starting task validation...")
    print(f"GPUs: {num_gpus}")
    print(f"Seeds per task: {args.num_seeds}")
    
    results = validate_all_tasks(
        levels=levels,
        task_paths=task_paths,
        num_gpus=num_gpus,
        num_seeds=args.num_seeds,
        verbose=args.verbose,
    )
    
    # Print summary
    print_summary(results)
    
    # Show fix snippets if requested
    if args.show_fixes and results['problematic_tasks']:
        print("\nFIX SNIPPETS")
        print("=" * 60)
        for task_name in results['problematic_tasks']:
            rec = results['recommendations'].get(task_name, {})
            snippet = generate_fix_snippet(task_name, rec)
            print(snippet)
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make results JSON-serializable
        serializable_results = {
            'summary': results['summary'],
            'problematic_tasks': results['problematic_tasks'],
            'recommendations': results['recommendations'],
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    # Exit with error code if there are problematic tasks
    if results['summary']['problematic'] > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
