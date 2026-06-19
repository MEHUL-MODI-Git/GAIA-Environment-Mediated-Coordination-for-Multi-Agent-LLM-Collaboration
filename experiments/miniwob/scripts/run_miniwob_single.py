#!/usr/bin/env python3
"""Single-agent MiniWoB++ baseline (Navigator + Verifier only, no Planner/Critic)

This is the ablation baseline for the GAIA MiniWoB++ experiment.
Strips out WebPlannerAgent and WebCriticAgent — leaving a single decision-making
agent (WebNavigator) plus the WebVerifier for success detection.

Usage:
    python experiments/miniwob/scripts/run_miniwob_single.py
    python experiments/miniwob/scripts/run_miniwob_single.py --tasks click-button
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    WebNavigatorAgent,
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


class Colors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    HEADER = "\033[95m"


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_progress(current, total, task_id, status):
    progress = f"[{current}/{total}]"
    print(f"{Colors.OKBLUE}{progress:>10} {task_id:35} {status}{Colors.ENDC}")


def print_stats(results):
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
    max_iterations: int = 25,
) -> dict:
    """Run single-agent baseline on one MiniWoB++ task"""
    task_id = task_spec["task_id"]

    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print_progress(task_idx + 1, total_tasks, task_id, "Starting (single-agent)...")
    print(f"  Instruction: {task_spec.get('instruction', '')[:70]}")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

    try:
        budget_monitor.reset()
        metrics = MetricsCollector()

        task_name = task_spec.get("task_name", task_id.split("/")[-1])
        task_log_dir = (log_dir or Path("logs")) / task_name
        mw_logger = MiniWoBLogger(
            task_id=task_id,
            log_dir=task_log_dir,
            run_log_path=run_log_path,
        )

        blackboard = Blackboard(logger=mw_logger.blackboard_logger)

        policy = Policy(
            max_iterations=max_iterations,
            branch_trigger_on_failure=False,
            verification_strictness="success_flag",
            stop_on_first_pass=True,
        )

        # Single-agent pool: Navigator + Verifier only (no Planner, no Critic)
        agents = [
            WebNavigatorAgent(
                name="WebNavigator",
                role="navigator",
                tier=ModelTier.SLOW,
                llm=slow_llm,
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

        print(f"{Colors.OKCYAN}▶ Running with 1 agent (Navigator + Verifier only)...{Colors.ENDC}\n")

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
        traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="Single-agent MiniWoB++ baseline")
    parser.add_argument("--tasks", nargs="+", help="Task names to run")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--output", type=str)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    print_header("Single-Agent MiniWoB++ Baseline (Navigator only)")

    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Colors.FAIL}❌ OPENAI_API_KEY not set!{Colors.ENDC}")
        return 1

    EXPERIMENT_DIR = Path(__file__).parent.parent
    tasks_path = PROJECT_ROOT / "data" / "miniwob" / "tasks.json"
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = results_dir / f"miniwob_single_{timestamp}.checkpoint.json"
    output_path = (Path(args.output) if args.output
                   else results_dir / f"miniwob_single_{timestamp}.results.json")

    logs_dir = EXPERIMENT_DIR / "logs" / f"single_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / "run_events.jsonl"
    print(f"Logs:    {logs_dir}")
    print(f"Output:  {output_path}\n")

    loader = MiniWoBLoader(tasks_path)
    if args.tasks:
        tasks = loader.load_by_names(args.tasks)
    elif args.difficulty:
        tasks = loader.load_by_difficulty(args.difficulty)
    else:
        tasks = loader.load()

    print(f"✓ Loaded {len(tasks)} tasks\n")

    fast_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model="gpt-4o", tier=ModelTier.SLOW)

    budget_monitor = BudgetMonitor(
        max_cost_per_problem=1.00,
        max_iterations=args.max_iterations,
        max_llm_calls=60,   # 2 agents × 30 iterations
    )

    checkpoint = CheckpointManager(checkpoint_path)
    checkpoint.start_run(total_problems=len(tasks))
    completed_ids = checkpoint.get_completed_task_ids() if not args.no_checkpoint else set()
    if completed_ids:
        print(f"{Colors.WARNING}Resuming: {len(completed_ids)} tasks already done{Colors.ENDC}")

    all_results = []

    try:
        for i, task_spec in enumerate(tasks):
            task_id = task_spec["task_id"]

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

            if (i + 1) % 5 == 0 or i == len(tasks) - 1:
                print_stats(all_results)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted — saving partial results{Colors.ENDC}")

    # Save results
    passed = sum(1 for r in all_results if r.get("passed"))
    total = len(all_results)
    summary = {
        "condition": "single_agent",
        "agents": ["WebNavigator", "WebVerifier"],
        "run_timestamp": timestamp,
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in all_results),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"Results saved to: {output_path}")
    print(f"Final: {passed}/{total} = {100*passed/total:.1f}% (single-agent baseline)")
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
