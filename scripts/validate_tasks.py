#!/usr/bin/env python3
"""
Task validation script for KernelBench.

Runs robustness filters on all tasks to identify problematic ones and
suggests comparison modes to fix them.

Usage:
    python scripts/validate_tasks.py --level 1
    python scripts/validate_tasks.py --level 1 2 3 4 5 6 7
    python scripts/validate_tasks.py --all
    python scripts/validate_tasks.py --task KernelBench/level1/23_Softmax.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.filters import (
    validate_task,
    get_recommended_comparison_mode,
)


def get_tasks_for_level(level: int) -> List[str]:
    """Get all task file paths for a given level."""
    level_dir = REPO_ROOT / "KernelBench" / f"level{level}"
    if not level_dir.exists():
        print(f"Warning: Level directory {level_dir} does not exist")
        return []
    
    tasks = sorted(level_dir.glob("*.py"))
    return [str(t) for t in tasks]


def validate_all_tasks(
    levels: List[int] = None,
    task_paths: List[str] = None,
    device: str = 'cuda',
    verbose: bool = False,
    num_seeds: int = 3,
) -> Dict[str, Any]:
    """
    Validate all tasks and return results.
    
    Args:
        levels: List of level numbers to validate
        task_paths: Specific task paths to validate (overrides levels)
        device: Device to run validation on
        verbose: Print progress
        num_seeds: Number of random seeds to use for each task
        
    Returns:
        Dict with validation results for each task
    """
    results = {
        'summary': {
            'total': 0,
            'problematic': 0,
            'passed': 0,
            'errors': 0,
        },
        'tasks': {},
        'problematic_tasks': [],
        'recommendations': {},
    }
    
    # Collect task paths
    all_tasks = []
    if task_paths:
        all_tasks = task_paths
    elif levels:
        for level in levels:
            all_tasks.extend(get_tasks_for_level(level))
    
    results['summary']['total'] = len(all_tasks)
    
    for i, task_path in enumerate(all_tasks):
        task_name = os.path.basename(task_path)
        
        if verbose:
            print(f"[{i+1}/{len(all_tasks)}] Validating {task_name}...")
        
        try:
            task_result = validate_task(task_path, device=device, num_seeds=num_seeds)
            results['tasks'][task_name] = task_result
            
            if task_result.get('error'):
                results['summary']['errors'] += 1
                if verbose:
                    print(f"  ERROR: {task_result['error']}")
            elif task_result.get('is_problematic'):
                results['summary']['problematic'] += 1
                results['problematic_tasks'].append(task_name)
                
                # Get recommended comparison mode
                recommended = get_recommended_comparison_mode(task_name)
                results['recommendations'][task_name] = {
                    'comparison_mode': recommended,
                    'filters_failed': [k for k, v in task_result['filters'].items() if v],
                    'suggestions': task_result.get('recommendations', []),
                }
                
                if verbose:
                    print(f"  PROBLEMATIC - recommend: {recommended}")
                    for suggestion in task_result.get('recommendations', []):
                        print(f"    - {suggestion}")
            else:
                results['summary']['passed'] += 1
                if verbose:
                    print(f"  OK")
                    
        except Exception as e:
            results['summary']['errors'] += 1
            results['tasks'][task_name] = {'error': str(e)}
            if verbose:
                print(f"  ERROR: {e}")
    
    return results


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
        '--device', '-d',
        type=str,
        default='cuda',
        help='Device to run validation on (default: cuda)'
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
    parser.add_argument(
        '--num-seeds', '-n',
        type=int,
        default=3,
        help='Number of random seeds to use for each task (default: 3)'
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
    
    # Run validation
    print("Starting task validation...")
    print(f"Device: {args.device}")
    print(f"Num seeds: {args.num_seeds}")
    
    results = validate_all_tasks(
        levels=levels,
        task_paths=task_paths,
        device=args.device,
        verbose=args.verbose,
        num_seeds=args.num_seeds,
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

