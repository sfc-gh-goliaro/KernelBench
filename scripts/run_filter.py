"""
Run validation filters on KernelBench tasks.

This script applies various validation filters to KernelBench tasks to identify
tasks that may have issues such as:
- Output values always in a small range
- Low output variance across random seeds
- Low variance along certain axes
- Inputs not affecting outputs
- Redundant or inefficient operations (via LLM analysis)

The script automatically uses all available GPUs for parallel processing when
filtering multiple tasks.

Usage:
    # Filter a single task
    python scripts/run_filter.py --task_path KernelBench/level1/1_Square_matrix_multiplication_.py
    
    # Filter all tasks in a level (uses all GPUs automatically)
    python scripts/run_filter.py --level 1
    
    # Filter with specific number of GPUs
    python scripts/run_filter.py --level 1 --num_gpus 4
    
    # Filter all tasks in a level with custom number of seeds
    python scripts/run_filter.py --level 1 --num_seeds 10
    
    # Skip LLM sanity check (faster)
    python scripts/run_filter.py --level 1 --skip_llm
    
    # Remove all filter result files
    python scripts/run_filter.py --clean
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple, Union
import multiprocessing as mp
from functools import partial

import litellm
import torch
from tqdm import tqdm

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Get repository root
REPO_ROOT = Path(__file__).parent.parent


# ============================================================================
# Task Loading
# ============================================================================

class KernelBenchTask:
    """Wrapper class to load and interact with KernelBench task files."""
    
    def __init__(self, task_path: str):
        """
        Initialize a KernelBenchTask from a task file path.
        
        Args:
            task_path: Path to the KernelBench task Python file
        """
        self.task_path = task_path
        self.task_name = os.path.basename(task_path)
        
        # Load the task module dynamically
        self.task_module = self._load_task_module(task_path)
        
        # Load the source code
        with open(task_path, 'r') as f:
            self.module_text = f.read()
    
    def _load_task_module(self, task_path: str):
        """Dynamically load a Python module from a file path."""
        module_name = os.path.splitext(os.path.basename(task_path))[0]
        module_name = f"kernelbench_task_{module_name}_{id(self)}"
        
        spec = importlib.util.spec_from_file_location(module_name, task_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load task module from {task_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def get_model(self):
        """Get an instance of the Model class from the task."""
        if not hasattr(self.task_module, 'Model'):
            raise AttributeError(f"Task {self.task_name} does not have a Model class")
        
        # Get initialization inputs if available
        if hasattr(self.task_module, 'get_init_inputs'):
            init_inputs = self.task_module.get_init_inputs()
            if init_inputs:
                return self.task_module.Model(*init_inputs)
        
        # Try to instantiate without arguments
        return self.task_module.Model()
    
    def get_inputs(self) -> List[Any]:
        """Get inputs for the task."""
        if not hasattr(self.task_module, 'get_inputs'):
            raise AttributeError(f"Task {self.task_name} does not have a get_inputs function")
        
        return self.task_module.get_inputs()


# ============================================================================
# Helper Functions
# ============================================================================

def set_seed(seed: int):
    """Set the seed for the random number generator."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def move_to_device(inputs, device: str):
    """Move inputs to the specified device."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        return [move_to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: move_to_device(v, device) for k, v in inputs.items()}
    else:
        return inputs


# ============================================================================
# LLM Functions
# ============================================================================

def llm_query(
    model_names: Union[str, List[str]],
    msg: str,
    system_message: str,
    temperatures: Union[float, List[float]] = 0.75,
    max_tokens: int = 4096,
    msg_history: List = None,
    print_debug: bool = False,
    verbose: bool = True,
) -> Tuple[str, List, str, float]:
    """Query the LLM using litellm."""
    if msg_history is None:
        msg_history = []
    
    # Select model
    if isinstance(model_names, list):
        import random
        model_name = random.choice(model_names)
    else:
        model_name = model_names
    
    # Select temperature
    if isinstance(temperatures, list):
        import random
        temp = random.choice(temperatures)
    else:
        temp = temperatures
    
    # Suppress verbose output
    # if verbose:
    #     print(f"==> Querying with model {model_name} & temp {temp}.")
    
    # Build message history
    messages = [{"role": "system", "content": system_message}]
    messages.extend(msg_history)
    messages.append({"role": "user", "content": msg})
    
    try:
        # Use litellm for API calls
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content
        
        # Update message history
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content}
        ]
        
        if print_debug:
            print()
            print("*" * 20 + " LLM START " + "*" * 20)
            for j, msg_item in enumerate(new_msg_history):
                print(f'{j}, {msg_item["role"]}: {msg_item["content"]}')
            print(content)
            print("*" * 21 + " LLM END " + "*" * 21)
            print()
        
        return content, new_msg_history, model_name, temp
        
    except Exception as e:
        print(f"Error querying LLM: {e}")
        raise


def analyze_pytorch_code(kernel_code: str) -> str:
    """Create a prompt for LLM to analyze PyTorch code."""
    return f"""Here is the PyTorch code:

{kernel_code}

Please provide a very careful analysis to determine if the given Model class contains redundant or inefficient operations.

Please structure your response as:
REDUNDANT_ANSWER: ###True### or ###False###
INEFFICIENT_ANSWER: ###True### or ###False###
"""


def extract_analysis_answers(llm_output: str) -> Tuple[bool, bool]:
    """Extract redundancy and inefficiency answers from LLM output."""
    redundant_pattern = r"REDUNDANT_ANSWER:\s*###(True|False)###"
    inefficient_pattern = r"INEFFICIENT_ANSWER:\s*###(True|False)###"

    redundant_match = re.search(redundant_pattern, llm_output)
    inefficient_match = re.search(inefficient_pattern, llm_output)

    if not redundant_match or not inefficient_match:
        raise ValueError("Could not find expected answer format in LLM output")

    is_redundant = redundant_match.group(1) == "True"
    is_inefficient = inefficient_match.group(1) == "True"

    return is_redundant, is_inefficient


def filter_llm_sanity(
    module_text: str,
    model_name: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> Tuple[bool, bool, str]:
    """Filter out tasks with redundant or inefficient operations using LLM analysis."""
    SYSTEM_PROMPT = """You are an expert PyTorch code engineer specializing in identifying redundant operations and inefficient operations in the provided code."""

    outputs = llm_query(
        model_names=model_name,
        msg=analyze_pytorch_code(module_text),
        system_message=SYSTEM_PROMPT,
        temperatures=temperature,
        max_tokens=max_tokens,
    )

    is_redundant, is_inefficient = extract_analysis_answers(outputs[0])
    return is_redundant, is_inefficient, outputs[0]


# ============================================================================
# Filtering Functions
# ============================================================================

def run_filters(task_path: str, num_seeds: int = 5, skip_llm: bool = False, gpu_id: int = 0) -> dict:
    """
    Run all validation filters on a single task.
    
    Args:
        task_path: Path to the task file
        num_seeds: Number of random seeds to use for testing
        skip_llm: Whether to skip LLM sanity check
        gpu_id: GPU device ID to use
        
    Returns:
        dict: Dictionary containing filter results
    """
    try:
        # Set the GPU device for this process
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        task = KernelBenchTask(task_path)
        
        # Create model once and share it across all filters
        model = task.get_model().to(device)
        
        # Pre-generate all inputs with different seeds
        all_inputs = []
        for seed in range(num_seeds):
            set_seed(seed)
            inputs = task.get_inputs()
            inputs = move_to_device(inputs, device)
            all_inputs.append(inputs)
        
        # Run forward passes with pre-generated inputs
        outputs_varying_seeds = []
        for inputs in all_inputs:
            with torch.no_grad():
                out = model(*inputs).float()
                torch.cuda.synchronize()
                outputs_varying_seeds.append(out)  # Keep on GPU
        
        # Stack outputs on GPU
        all_outputs = torch.stack(outputs_varying_seeds)
        
        # Filter check 1: Output range not only in (-0.01, 0.01) - do on GPU
        should_filter_output_range = ((all_outputs > -0.01) & (all_outputs < 0.01)).all()
        should_filter_output_range = bool(should_filter_output_range.cpu())
        
        # Filter check 2: Output std not close to 0 - do on GPU
        stds = torch.std(all_outputs, dim=0)
        should_filter_output_std = (stds < 0.01).all()
        should_filter_output_std = bool(should_filter_output_std.cpu())
        
        # Filter check 3: Output axes variation not close to 0 - do on GPU
        axis_stds = []
        for axis in range(all_outputs.ndim):
            axis_stds.append(torch.std(all_outputs, dim=axis))
        should_filter_output_axes = any((std < 0.01).all() for std in axis_stds)
        should_filter_output_axes = bool(should_filter_output_axes)
        
        # Clean up axis_stds
        del axis_stds
        
        # Filter check 4: Input variation doesn't affect the output (reuse inputs from above)
        # Free memory from previous outputs
        del all_outputs, outputs_varying_seeds
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reuse the already generated inputs and run forward passes
        outputs_varying_inputs = []
        for inputs in all_inputs:
            with torch.no_grad():
                out = model(*inputs).float()
                torch.cuda.synchronize()
                outputs_varying_inputs.append(out)  # Keep on GPU
        
        all_outputs_varying_inputs = torch.stack(outputs_varying_inputs)
        stds = torch.std(all_outputs_varying_inputs, dim=0)
        should_filter_input_impact = (stds < 0.01).all()
        should_filter_input_impact = bool(should_filter_input_impact.cpu())
        
        # Clean up
        del all_outputs_varying_inputs, outputs_varying_inputs, all_inputs, stds
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Filter check 5: LLM sanity check
        if skip_llm:
            is_redundant = False
            is_inefficient = False
            llm_output = "Skipped"
        else:
            is_redundant, is_inefficient, llm_output = filter_llm_sanity(
                task.module_text,
                model_name="claude-3-7-sonnet-20250219",
                temperature=0.0,
                max_tokens=8192,
            )
        
        results = {
            "task_path": task_path,
            "task_name": os.path.basename(task_path),
            "filter_output_range": should_filter_output_range,
            "filter_output_std": should_filter_output_std,
            "filter_output_axes": should_filter_output_axes,
            "filter_input_impact": should_filter_input_impact,
            "filter_llm_redundancy": is_redundant,
            "filter_llm_inefficiency": is_inefficient,
            "filter_llm_assessment": llm_output,
        }
        
        # Determine if task should be filtered overall
        results["should_filter"] = (
            should_filter_output_range or
            should_filter_output_std or
            should_filter_output_axes or
            should_filter_input_impact or
            is_redundant or
            is_inefficient
        )
        
        # Clean up model and GPU memory before returning
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return results
        
    except Exception as e:
        # Clean up GPU memory even on error
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return {
            "task_path": task_path,
            "task_name": os.path.basename(task_path),
            "error": str(e),
            "should_filter": None,
        }


def get_level_tasks(level: int) -> List[str]:
    """Get all task files for a given level."""
    level_dir = REPO_ROOT / "KernelBench" / f"level{level}"
    if not level_dir.exists():
        raise ValueError(f"Level directory does not exist: {level_dir}")
    
    tasks = sorted(level_dir.glob("*.py"))
    return [str(task) for task in tasks]


def run_filters_worker(gpu_id, task_queue, result_queue, num_seeds, skip_llm):
    """
    Worker function for parallel processing.
    Each worker is assigned to a specific GPU and pulls tasks from a shared queue.
    
    Args:
        gpu_id: GPU device ID to use
        task_queue: Multiprocessing queue containing tasks to process
        result_queue: Multiprocessing queue to put results into
        num_seeds: Number of random seeds to use
        skip_llm: Whether to skip LLM sanity check
    """
    import queue
    
    while True:
        try:
            # Get next task from queue (blocking)
            task_path = task_queue.get()
            
            if task_path is None:  # Poison pill to signal worker to stop
                break
            
            # Process the task
            result = run_filters(task_path, num_seeds, skip_llm, gpu_id)
            
            # Put result in result queue
            result_queue.put(result)
            
        except Exception as e:
            # Put error result in queue with task path if available
            result_queue.put({
                "task_path": task_path if 'task_path' in locals() else "unknown",
                "task_name": os.path.basename(task_path) if 'task_path' in locals() else "unknown",
                "error": str(e),
                "should_filter": None,
            })


def clean_all_filter_results():
    """Remove all filter result files from the KernelBench directory."""
    total_removed = 0
    
    # Clean all levels
    for level in [1, 2, 3, 4]:
        level_dir = REPO_ROOT / "KernelBench" / f"level{level}"
        if level_dir.exists():
            pattern = str(level_dir / "*.filter_results.json")
            result_files = glob.glob(pattern)
            for result_file in result_files:
                Path(result_file).unlink()
                total_removed += 1
        
        # Remove combined results file if it exists
        combined_file = REPO_ROOT / f"level{level}_filter_results.json"
        if combined_file.exists():
            combined_file.unlink()
            total_removed += 1
    
    if total_removed > 0:
        print(f"Removed {total_removed} filter result file(s)")
    else:
        print("No filter result files found")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run validation filters on KernelBench tasks"
    )
    parser.add_argument(
        "--task_path",
        type=str,
        help="Path to a single task file to filter"
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        help="Level number to filter all tasks in that level"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of random seeds to use for testing (default: 5)"
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM sanity check (faster)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save filter results (default: same as task file)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all existing filter result files and exit (cannot be combined with other flags)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for parallel processing (default: all available GPUs)"
    )
    
    args = parser.parse_args()
    
    # Handle clean mode
    if args.clean:
        # Clean mode cannot be combined with other flags
        if args.task_path or args.level or args.skip_llm or args.output_dir or args.num_seeds != 5:
            parser.error("--clean cannot be combined with other flags")
        
        print("Cleaning all filter result files...")
        clean_all_filter_results()
        return
    
    # Validate arguments for normal filtering mode
    if not args.task_path and not args.level:
        parser.error("Either --task_path or --level must be specified")
    
    if args.task_path and args.level:
        parser.error("Cannot specify both --task_path and --level")
    
    # Get list of tasks to filter
    if args.task_path:
        tasks = [args.task_path]
    else:
        tasks = get_level_tasks(args.level)
        print(f"Found {len(tasks)} tasks in level {args.level}")
    
    # Determine number of GPUs to use
    num_available_gpus = torch.cuda.device_count()
    if args.num_gpus is None:
        num_gpus = num_available_gpus
    else:
        num_gpus = min(args.num_gpus, num_available_gpus)
    
    if num_gpus == 0:
        print("ERROR: No GPUs available!")
        return
    
    print(f"Using {num_gpus} GPU(s) for parallel processing")
    
    # Run filters on all tasks (parallel if multiple GPUs)
    print(f"\nFiltering {len(tasks)} task(s) using {num_gpus} GPU(s)...")
    
    if num_gpus == 1 or len(tasks) == 1:
        # Single GPU or single task - run sequentially with progress bar
        all_results = []
        for task_path in tqdm(tasks, desc="Filtering tasks", unit="task"):
            results = run_filters(task_path, args.num_seeds, args.skip_llm, gpu_id=0)
            all_results.append(results)
    else:
        # Multiple GPUs - use a shared task queue
        # Each GPU worker pulls tasks one at a time from the queue
        # This ensures GPUs stay busy and only one task runs per GPU at a time
        
        # Create queues for task distribution and result collection
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Fill task queue with all tasks
        for task_path in tasks:
            task_queue.put(task_path)
        
        # Add poison pills to signal workers to stop (one per worker)
        for _ in range(num_gpus):
            task_queue.put(None)
        
        # Start worker processes (one per GPU)
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=run_filters_worker,
                args=(gpu_id, task_queue, result_queue, args.num_seeds, args.skip_llm)
            )
            p.start()
            processes.append(p)
        
        # Collect results with progress bar
        all_results = []
        with tqdm(total=len(tasks), desc="Filtering tasks", unit="task") as pbar:
            for _ in range(len(tasks)):
                result = result_queue.get()
                all_results.append(result)
                pbar.update(1)
        
        # Wait for all workers to finish
        for p in processes:
            p.join()
        
        # Reorder results to match original task order
        result_map = {r['task_path']: r for r in all_results}
        all_results = [result_map[task_path] for task_path in tasks]
    
    # Save results only for tasks that should be filtered or have errors
    for i, results in enumerate(all_results):
        # Only save if task should be filtered or has an error
        if not (results.get("should_filter", False) or "error" in results):
            continue
            
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{results['task_name']}.filter_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Save next to the task file
            task_path = tasks[i]
            task_dir = Path(task_path).parent
            output_path = task_dir / f"{results['task_name']}.filter_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_tasks = len(all_results)
    error_tasks = sum(1 for r in all_results if "error" in r)
    filtered_tasks = sum(1 for r in all_results if r.get("should_filter", False))
    passed_tasks = total_tasks - error_tasks - filtered_tasks
    
    print(f"Total tasks: {total_tasks}")
    print(f"Passed: {passed_tasks} ({passed_tasks/total_tasks*100:.1f}%)")
    print(f"Should filter: {filtered_tasks} ({filtered_tasks/total_tasks*100:.1f}%)")
    print(f"Errors: {error_tasks} ({error_tasks/total_tasks*100:.1f}%)")
    
    # Breakdown by filter type
    if filtered_tasks > 0:
        print("\nFilter breakdown:")
        filter_counts = {
            "output_range": sum(1 for r in all_results if r.get("filter_output_range", False)),
            "output_std": sum(1 for r in all_results if r.get("filter_output_std", False)),
            "output_axes": sum(1 for r in all_results if r.get("filter_output_axes", False)),
            "input_impact": sum(1 for r in all_results if r.get("filter_input_impact", False)),
            "llm_redundancy": sum(1 for r in all_results if r.get("filter_llm_redundancy", False)),
            "llm_inefficiency": sum(1 for r in all_results if r.get("filter_llm_inefficiency", False)),
        }
        for filter_name, count in filter_counts.items():
            if count > 0:
                print(f"  {filter_name}: {count} tasks")
    
    # Save combined results if filtering multiple tasks
    # Only include tasks that should be filtered or have errors
    if len(tasks) > 1:
        filtered_results = [
            r for r in all_results 
            if r.get("should_filter", False) or "error" in r
        ]
        
        if args.output_dir:
            combined_output = Path(args.output_dir) / "all_filter_results.json"
        else:
            combined_output = REPO_ROOT / f"level{args.level}_filter_results.json"
        
        with open(combined_output, 'w') as f:
            json.dump(filtered_results, f, indent=4)
        print(f"\nSaved {len(filtered_results)} filtered task(s) to: {combined_output}")


if __name__ == "__main__":
    main()
