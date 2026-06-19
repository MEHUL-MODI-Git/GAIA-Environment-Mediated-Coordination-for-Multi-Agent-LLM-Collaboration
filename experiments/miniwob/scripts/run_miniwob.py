#!/usr/bin/env python3
"""Run GAIA on MiniWoB++ tasks with checkpoint support

Usage:
    python experiments/miniwob/scripts/run_miniwob.py
    python experiments/miniwob/scripts/run_miniwob.py --tasks click-button focus-text
    python experiments/miniwob/scripts/run_miniwob.py --difficulty easy

Supports:
- Automatic checkpointing after each task
- Resume from checkpoint after crash
- Filter tasks by name or difficulty
- Real-time progress display
- Success rate reporting
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env from project root (so OPENAI_API_KEY never needs to be set manually)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy
from gaia.agents.miniwob import (
    WebPlannerAgent,
    WebNavigatorAgent,
    WebCriticAgent,
    WebVerifierAgent,
)
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.web_loop import WebEpisodeLoop
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager
from gaia.utils.miniwob_logger import MiniWoBLogger
from gaia.benchmarks.miniwob.loader import MiniWoBLoader
from gaia.benchmarks.miniwob.evaluator import MiniWoBEvaluator


# Terminal colors
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_progress(current: int, total: int, task_id: str, status: str):
    progress = f"[{current}/{total}]"
    print(f"{Colors.OKBLUE}{progress:>10} {task_id:35} {status}{Colors.ENDC}")


def print_stats(results: List[dict]):
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    rate = passed / total if total > 0 else 0.0
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)

    print(f"\n{Colors.BOLD}Current statistics:{Colors.ENDC}")
    print(f"  Completed: {total}")
    print(f"  Passed:    {Colors.OKGREEN}{passed}{Colors.ENDC}")
    print(f"  Failed:    {Colors.FAIL}{total - passed}{Colors.ENDC}")
    print(f"  Rate:      {Colors.BOLD}{rate:.1%}{Colors.ENDC}")
    print(f"  Cost:      ${total_cost:.4f}\n")


async def run_single_task(
    task_spec: dict,
    task_idx: int,
    total_tasks: int,
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    budget_monitor: BudgetMonitor,
    log_dir: Path = None,
    run_log_path: Path = None,
    max_steps: int = 15,
    max_iterations: int = 10,
) -> dict:
    """Run GAIA on a single MiniWoB++ task"""
    task_id = task_spec["task_id"]

    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print_progress(task_idx + 1, total_tasks, task_id, "Starting...")
    print(f"  Instruction: {task_spec.get('instruction', '')[:70]}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

    try:
        budget_monitor.reset()
        metrics = MetricsCollector()

        # Per-task structured logger
        task_name = task_spec.get("task_name", task_id.split("/")[-1])
        task_log_dir = (log_dir or Path("logs")) / task_name
        mw_logger = MiniWoBLogger(
            task_id=task_id,
            log_dir=task_log_dir,
            run_log_path=run_log_path,
        )

        # Fresh blackboard per task
        blackboard = Blackboard(logger=mw_logger.blackboard_logger)

        policy = Policy(
            max_iterations=max_iterations,
            branch_trigger_on_failure=False,
            verification_strictness="success_flag",
            stop_on_first_pass=True,
        )

        # Build agent pool: 4 agents
        # - WebPlanner: plans action sequence (fires once at start, replans on strategy reset)
        # - WebNavigator: picks next action each iteration (backs off on CONFLICT)
        # - WebCritic: fires when Navigator is stuck, posts feedback, resolves CONFLICT
        # - WebVerifier: checks success_flag, raises CONFLICT when stuck too long
        # (DOMAnalyzer dropped — Navigator reads structured elements from task metadata directly)
        agents = [
            WebPlannerAgent(
                name="WebPlanner",
                role="planner",
                tier=ModelTier.SLOW,
                llm=slow_llm,
                blackboard=blackboard,
                metrics=metrics,
                budget_monitor=budget_monitor,
            ),
            WebNavigatorAgent(
                name="WebNavigator",
                role="navigator",
                tier=ModelTier.SLOW,
                llm=slow_llm,
                blackboard=blackboard,
                metrics=metrics,
                budget_monitor=budget_monitor,
            ),
            WebCriticAgent(
                name="WebCritic",
                role="critic",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                metrics=metrics,
                budget_monitor=budget_monitor,
            ),
            WebVerifierAgent(
                name="WebVerifier",
                role="verifier",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                metrics=metrics,
                budget_monitor=budget_monitor,
            ),
        ]

        # Override max_steps from task spec
        task_spec_copy = dict(task_spec)
        task_spec_copy["max_steps"] = max(task_spec.get("max_steps", 15), max_steps)

        loop = WebEpisodeLoop(
            blackboard=blackboard,
            agents=agents,
            metrics=metrics,
            policy=policy,
            budget_monitor=budget_monitor,
            miniwob_logger=mw_logger,
        )

        print(f"{Colors.OKCYAN}▶ Running with 4 agents "
              f"(Planner, Navigator, Critic, Verifier)...{Colors.ENDC}\n")

        start_time = asyncio.get_event_loop().time()
        result = await loop.run_episode(task_spec_copy)
        duration_s = asyncio.get_event_loop().time() - start_time

        cost = budget_monitor.current_cost
        if result.passed:
            status = (f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC} "
                      f"({result.steps_taken} steps, ${cost:.4f}, {duration_s:.1f}s)")
        else:
            status = (f"{Colors.FAIL}✗ FAILED{Colors.ENDC} "
                      f"({result.steps_taken} steps, ${cost:.4f}, {duration_s:.1f}s)")

        print_progress(task_idx + 1, total_tasks, task_id, status)

        return {
            "task_id": task_id,
            "task_name": task_spec.get("task_name", ""),
            "difficulty": task_spec.get("difficulty", "unknown"),
            "passed": result.passed,
            "steps_taken": result.steps_taken,
            "cost_usd": cost,
            "duration_s": duration_s,
            "log_dir": str(task_log_dir),
            "error": None,
        }

    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print_progress(task_idx + 1, total_tasks, task_id,
                       f"{Colors.WARNING}⚠ ERROR: {error_msg}{Colors.ENDC}")
        return {
            "task_id": task_id,
            "task_name": task_spec.get("task_name", ""),
            "difficulty": task_spec.get("difficulty", "unknown"),
            "passed": False,
            "steps_taken": 0,
            "cost_usd": 0.0,
            "duration_s": 0.0,
            "error": error_msg,
        }


async def main():
    parser = argparse.ArgumentParser(description="Run GAIA on MiniWoB++ tasks")
    parser.add_argument("--tasks", nargs="+", help="Task names to run (e.g. click-button)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--max-steps", type=int, default=15, help="Max browser steps per task")
    parser.add_argument("--max-iterations", type=int, default=25, help="Max episode iterations")
    parser.add_argument("--output", type=str, help="Output results JSON path")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    args = parser.parse_args()

    print_header("GAIA MiniWoB++ Benchmark")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Colors.FAIL}❌ OPENAI_API_KEY not set!{Colors.ENDC}")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1

    # Paths (relative to this script: scripts/ -> miniwob/ -> experiments/)
    EXPERIMENT_DIR = Path(__file__).parent.parent
    tasks_path = PROJECT_ROOT / "data" / "miniwob" / "tasks.json"
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = results_dir / f"miniwob_{timestamp}.checkpoint.json"
    output_path = Path(args.output) if args.output else results_dir / f"miniwob_{timestamp}.results.json"

    # Per-task and run-level logs
    logs_dir = EXPERIMENT_DIR / "logs" / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / "run_events.jsonl"
    print(f"Logs:         {logs_dir}")

    # Load tasks
    print(f"Loading tasks from {tasks_path}...")
    loader = MiniWoBLoader(tasks_path)

    if args.tasks:
        tasks = loader.load_by_names(args.tasks)
        if not tasks:
            print(f"{Colors.WARNING}No tasks found for names: {args.tasks}{Colors.ENDC}")
            return 1
    elif args.difficulty:
        tasks = loader.load_by_difficulty(args.difficulty)
    else:
        tasks = loader.load()

    print(f"✓ Loaded {len(tasks)} tasks\n")

    # Initialize LLMs
    print("Initializing LLMs...")
    fast_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model="gpt-4o", tier=ModelTier.SLOW)
    print("✓ LLMs initialized\n")

    # Budget monitor
    budget_monitor = BudgetMonitor(
        max_cost_per_problem=1.50,   # enough for hard tasks + strategy reset
        max_iterations=args.max_iterations,
        max_llm_calls=120,           # 4 agents × 30 iterations
    )

    # Checkpoint
    checkpoint = CheckpointManager(checkpoint_path)
    checkpoint.start_run(total_problems=len(tasks))
    completed_ids = checkpoint.get_completed_task_ids() if not args.no_checkpoint else set()
    if completed_ids:
        print(f"{Colors.WARNING}Resuming: {len(completed_ids)} tasks already done{Colors.ENDC}")

    all_results = []

    try:
        for i, task_spec in enumerate(tasks):
            task_id = task_spec["task_id"]

            # Skip completed
            if task_id in completed_ids:
                print_progress(i + 1, len(tasks), task_id,
                               f"{Colors.OKBLUE}(skipped — already done){Colors.ENDC}")
                continue

            result = await run_single_task(
                task_spec=task_spec,
                task_idx=i,
                total_tasks=len(tasks),
                fast_llm=fast_llm,
                slow_llm=slow_llm,
                budget_monitor=budget_monitor,
                log_dir=logs_dir,
                run_log_path=run_log_path,
                max_steps=args.max_steps,
                max_iterations=args.max_iterations,
            )
            all_results.append(result)

            # Checkpoint
            if not args.no_checkpoint:
                checkpoint.add_result(
                    task_id=task_id,
                    passed=result["passed"],
                    iterations=result["steps_taken"],
                    cost_usd=result["cost_usd"],
                    duration_s=result["duration_s"],
                    stop_reason="passed" if result["passed"] else "failed",
                    error=result.get("error"),
                )

            # Print running stats every 5 tasks
            if (i + 1) % 5 == 0:
                print_stats(all_results)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted! Saving partial results...{Colors.ENDC}")

    # Final evaluation
    evaluator = MiniWoBEvaluator()
    metrics = evaluator.evaluate(all_results)
    evaluator.print_summary(metrics)

    # Save results
    output_data = {
        "run_timestamp": timestamp,
        "total_tasks": len(tasks),
        "metrics": metrics,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")
    print(f"Episode logs:  {logs_dir}/")
    print(f"  Per-task:    {logs_dir}/<task-name>/episode.{{jsonl,txt}}")
    print(f"  Run-level:   {run_log_path}")

    return 0 if metrics["success_rate"] > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
