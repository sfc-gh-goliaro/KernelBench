#!/usr/bin/env python3
"""
Migration script to update KernelBench task files with TASK_CONFIG and configurable get_inputs.

This script:
1. Adds TASK_CONFIG dict to each task file
2. Updates get_inputs() to accept **kwargs for configurable dimensions
3. Sets appropriate comparison_mode for problematic tasks

Usage:
    python scripts/migrate_tasks.py --level 1
    python scripts/migrate_tasks.py --all
    python scripts/migrate_tasks.py --dry-run --level 1
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent

# Tasks that need special comparison modes based on robust-kbench analysis
SPECIAL_COMPARISON_MODES = {
    # Level 1
    '23_Softmax.py': 'log_domain',
    '24_LogSoftmax.py': 'default',  # Already in log domain
    '37_FrobeniusNorm_.py': 'relative',
    '38_L1Norm_.py': 'relative',
    '39_L2Norm_.py': 'relative',
    '36_RMSNorm_.py': 'relative',
    '40_LayerNorm.py': 'relative',
    '33_BatchNorm.py': 'relative',
    '34_InstanceNorm.py': 'relative',
    '35_GroupNorm_.py': 'relative',
    '90_cumprod.py': 'log_domain',
    '98_KLDivLoss.py': 'relative',
    '94_MSELoss.py': 'relative',
    '95_CrossEntropyLoss.py': 'relative',
    '96_HuberLoss.py': 'relative',
    '99_TripletMarginLoss.py': 'relative',
    '100_HingeLoss.py': 'relative',
    
    # Level 2 - tasks ending in softmax
    '66_Matmul_Dropout_Softmax.py': 'topk',
    '84_Gemm_BatchNorm_Scaling_Softmax.py': 'log_domain',
    '99_Matmul_GELU_Softmax.py': 'log_domain',
    '49_ConvTranspose3d_Softmax_Sigmoid.py': 'distribution',
    '91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid.py': 'distribution',
    '6_Conv3d_Softmax_MaxPool_MaxPool.py': 'log_domain',
    
    # Level 2 - tasks with multiple means/GAPs
    '44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean.py': 'relative',
    '42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py': 'relative',
}


def get_comparison_mode(filename: str) -> str:
    """Determine the appropriate comparison mode for a task."""
    basename = os.path.basename(filename)
    
    # Check explicit mappings first
    if basename in SPECIAL_COMPARISON_MODES:
        return SPECIAL_COMPARISON_MODES[basename]
    
    # Heuristic-based detection
    name_lower = basename.lower()
    
    # Softmax at the end of the name (but not LogSoftmax)
    if name_lower.endswith('softmax.py') and 'log' not in name_lower:
        return 'log_domain'
    
    # Normalization operations
    if any(x in name_lower for x in ['norm', 'l1norm', 'l2norm', 'frobenius', 'rms']):
        return 'relative'
    
    # Loss functions
    if 'loss' in name_lower:
        return 'relative'
    
    # Cumulative product
    if 'cumprod' in name_lower:
        return 'log_domain'
    
    return 'default'


def extract_global_vars(content: str) -> Dict[str, str]:
    """Extract global variable assignments from the file content."""
    vars_dict = {}
    
    # Match simple variable assignments at module level
    # e.g., batch_size = 128
    pattern = r'^(\w+)\s*=\s*(.+?)(?:\s*#.*)?$'
    
    for line in content.split('\n'):
        # Skip if inside a function or class
        if line.startswith(' ') or line.startswith('\t'):
            continue
        match = re.match(pattern, line.strip())
        if match:
            var_name, var_value = match.groups()
            # Skip class and function definitions, imports
            if var_name not in ('class', 'def', 'import', 'from', 'Model'):
                vars_dict[var_name] = var_value.strip()
    
    return vars_dict


def find_get_inputs_function(content: str) -> Tuple[int, int, str]:
    """Find the get_inputs function in the content.
    
    Returns (start_line, end_line, function_text)
    """
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if line.startswith('def get_inputs('):
            start_idx = i
        elif start_idx is not None and (line.startswith('def ') or line.startswith('class ') or 
                                         (line.strip() and not line.startswith(' ') and not line.startswith('\t'))):
            end_idx = i
            break
    
    if start_idx is not None and end_idx is None:
        end_idx = len(lines)
    
    if start_idx is not None:
        return start_idx, end_idx, '\n'.join(lines[start_idx:end_idx])
    
    return -1, -1, ''


def create_task_config(global_vars: Dict[str, str], comparison_mode: str) -> str:
    """Create the TASK_CONFIG dict string."""
    # Filter to only include relevant input dimension variables
    relevant_vars = {}
    for name, value in global_vars.items():
        # Include common dimension-related variables
        if any(x in name.lower() for x in ['batch', 'dim', 'size', 'shape', 'features', 
                                            'channels', 'height', 'width', 'length',
                                            'seq', 'hidden', 'num_', 'in_', 'out_']):
            relevant_vars[name] = value
    
    lines = ["TASK_CONFIG = {"]
    
    # Add comparison mode
    lines.append(f"    'comparison_mode': '{comparison_mode}',")
    
    # Add tolerance settings
    if comparison_mode in ('relative', 'log_domain'):
        lines.append("    'atol': 1e-5,")
        lines.append("    'rtol': 1e-3,")
    else:
        lines.append("    'atol': 1e-4,")
        lines.append("    'rtol': 1e-4,")
    
    # Add topk_k for topk mode
    if comparison_mode == 'topk':
        lines.append("    'topk_k': 10,")
    
    lines.append("}")
    
    return '\n'.join(lines)


def update_get_inputs(original_func: str, global_vars: Dict[str, str]) -> str:
    """Update get_inputs function to accept **kwargs."""
    lines = original_func.split('\n')
    
    # Check if already has kwargs
    if '**kwargs' in lines[0] or '**' in lines[0]:
        return original_func  # Already updated
    
    # Update function signature
    if lines[0].strip() == 'def get_inputs():':
        lines[0] = 'def get_inputs(**kwargs):'
    elif 'def get_inputs(' in lines[0]:
        # Has parameters, add **kwargs at the end
        lines[0] = re.sub(r'\):', ', **kwargs):', lines[0])
    
    # Add docstring if not present
    if len(lines) > 1 and '"""' not in lines[1]:
        indent = '    '
        docstring = [
            f'{indent}"""',
            f'{indent}Generate inputs for the model.',
            f'{indent}',
            f'{indent}Args:',
            f'{indent}    **kwargs: Override default dimensions (e.g., batch_size=32)',
            f'{indent}',
            f'{indent}Returns:',
            f'{indent}    List of input tensors',
            f'{indent}"""',
        ]
        lines = [lines[0]] + docstring + lines[1:]
    
    return '\n'.join(lines)


def migrate_file(filepath: str, dry_run: bool = False) -> Tuple[bool, str]:
    """Migrate a single task file.
    
    Returns (success, message)
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if already migrated
        if 'TASK_CONFIG' in content:
            return True, "Already migrated"
        
        filename = os.path.basename(filepath)
        comparison_mode = get_comparison_mode(filename)
        global_vars = extract_global_vars(content)
        
        # Create TASK_CONFIG
        task_config = create_task_config(global_vars, comparison_mode)
        
        # Find where to insert TASK_CONFIG (after imports, before class definition)
        lines = content.split('\n')
        insert_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif line.startswith('class ') and insert_idx > 0:
                break
        
        # Find empty line after imports for clean insertion
        while insert_idx < len(lines) and lines[insert_idx].strip() == '':
            insert_idx += 1
        
        # Insert TASK_CONFIG
        lines.insert(insert_idx, '')
        lines.insert(insert_idx + 1, task_config)
        lines.insert(insert_idx + 2, '')
        
        new_content = '\n'.join(lines)
        
        # Update get_inputs function
        start, end, func_text = find_get_inputs_function(new_content)
        if start >= 0 and func_text:
            new_func = update_get_inputs(func_text, global_vars)
            lines = new_content.split('\n')
            lines[start:end] = new_func.split('\n')
            new_content = '\n'.join(lines)
        
        if dry_run:
            return True, f"Would add TASK_CONFIG with comparison_mode='{comparison_mode}'"
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        return True, f"Added TASK_CONFIG with comparison_mode='{comparison_mode}'"
        
    except Exception as e:
        return False, f"Error: {e}"


def get_tasks_for_level(level: int) -> List[str]:
    """Get all task file paths for a given level."""
    level_dir = REPO_ROOT / "KernelBench" / f"level{level}"
    if not level_dir.exists():
        return []
    
    tasks = sorted(level_dir.glob("*.py"))
    return [str(t) for t in tasks]


def main():
    parser = argparse.ArgumentParser(
        description="Migrate KernelBench tasks to use TASK_CONFIG"
    )
    parser.add_argument(
        '--level', '-l',
        type=int,
        nargs='+',
        help='Level(s) to migrate (1-7)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Migrate all levels (1-7)'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        nargs='+',
        help='Specific task file(s) to migrate'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Collect tasks to migrate
    tasks = []
    if args.task:
        tasks = args.task
    elif args.all:
        for level in range(1, 8):
            tasks.extend(get_tasks_for_level(level))
    elif args.level:
        for level in args.level:
            tasks.extend(get_tasks_for_level(level))
    else:
        print("Error: Must specify --level, --all, or --task")
        parser.print_help()
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN - No changes will be made\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for task in tasks:
        filename = os.path.basename(task)
        success, message = migrate_file(task, dry_run=args.dry_run)
        
        if success:
            if "Already migrated" in message:
                skip_count += 1
                if args.verbose:
                    print(f"SKIP: {filename} - {message}")
            else:
                success_count += 1
                print(f"OK: {filename} - {message}")
        else:
            error_count += 1
            print(f"ERROR: {filename} - {message}")
    
    print(f"\nSummary: {success_count} migrated, {skip_count} skipped, {error_count} errors")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

