#!/usr/bin/env python3
"""Run GAIA on all HumanEval problems with checkpoint and logging support

Supports:
- Automatic checkpointing after each problem (atomic writes)
- Resume from checkpoint after crash/interrupt (skips completed problems)
- Per-problem event logs in results/logs/<task_id>.jsonl
- Clean terminal output (no event spam, only progress lines)
- Error isolation (single failure doesn't stop run)
- Progress tracking with pass rate
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy, SignalType
from gaia.blackboard.storage import InMemoryStorage
from gaia.agents.coder import CoderAgent
from gaia.agents.critic import CriticAgent
from gaia.agents.verifier import VerifierAgent
from gaia.agents.edge_case import EdgeCaseAgent
from gaia.llms.openai_llm import OpenAILLM, ModelTier
from gaia.episode.loop import EpisodeLoop
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager
from gaia.utils.metrics import MetricsCollector
from gaia.utils.blackboard_logger import BlackboardLogger

# Suppress verbose agent/episode Python logs — all events are in per-problem .jsonl files
import logging
logging.disable(logging.INFO)


# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_progress(current: int, total: int, task_id: str, status: str, color: str = Colors.OKBLUE):
    """Print progress line"""
    progress = f"[{current}/{total}]"
    print(f"{color}{progress:>10} {task_id:20} {status}{Colors.ENDC}", flush=True)


def print_stats(checkpoint: CheckpointManager):
    """Print current statistics"""
    pass_rate = checkpoint.pass_rate * 100
    cost = checkpoint.total_cost_usd

    print(f"\n{Colors.BOLD}Statistics:{Colors.ENDC}")
    print(f"  Completed: {checkpoint.completed_count}/{checkpoint.total_problems}")
    print(f"  Passed:    {Colors.OKGREEN}{checkpoint.passed_count}{Colors.ENDC}")
    print(f"  Failed:    {Colors.FAIL}{checkpoint.failed_count}{Colors.ENDC}")
    print(f"  Errors:    {Colors.WARNING}{checkpoint.error_count}{Colors.ENDC}")
    print(f"  Pass Rate: {Colors.BOLD}{pass_rate:.1f}%{Colors.ENDC}")
    print(f"  Cost:      ${cost:.4f}")
    print()


async def run_single_problem(
    problem: dict,
    problem_idx: int,
    total_problems: int,
    agents: List,
    blackboard: Blackboard,
    budget_monitor: BudgetMonitor,
    max_iterations: int = 10,
) -> dict:
    """Run GAIA on a single HumanEval problem

    Returns:
        dict with keys: passed, iterations, cost_usd, duration_s, stop_reason,
                       num_conflicts, error (if crashed)
    """
    task_id = problem["task_id"]

    # Print start
    print_progress(problem_idx + 1, total_problems, task_id, "Starting...", Colors.OKCYAN)

    try:
        # Reset budget for this problem
        budget_monitor.reset()

        # Create episode loop
        policy = Policy(max_iterations=max_iterations)
        metrics = MetricsCollector()
        loop = EpisodeLoop(
            blackboard=blackboard,
            agents=agents,
            metrics=metrics,
            policy=policy,
            budget_monitor=budget_monitor,
        )

        # Run episode
        start_time = asyncio.get_event_loop().time()
        result = await loop.run_episode(problem)
        duration_s = asyncio.get_event_loop().time() - start_time

        # Get conflict count from blackboard
        num_conflicts = len(blackboard.get_signals(signal_type=SignalType.CONFLICT))

        # Determine stop reason and status message
        if result.passed:
            stop_reason = "passed"
            status_msg = (
                f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC} "
                f"({result.iterations} iter, ${budget_monitor.current_cost:.4f}, {duration_s:.1f}s)"
            )
        else:
            stop_reason = "max_iterations" if result.iterations >= max_iterations else "failed"
            status_msg = (
                f"{Colors.FAIL}✗ FAILED{Colors.ENDC} "
                f"({result.iterations} iter, ${budget_monitor.current_cost:.4f}, {duration_s:.1f}s)"
            )

        print_progress(problem_idx + 1, total_problems, task_id, status_msg)

        return {
            "passed": result.passed,
            "iterations": result.iterations,
            "cost_usd": budget_monitor.current_cost,
            "duration_s": duration_s,
            "stop_reason": stop_reason,
            "num_conflicts": num_conflicts,
            "num_branches": 0,
            "error": None,
        }

    except KeyboardInterrupt:
        raise

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print_progress(
            problem_idx + 1,
            total_problems,
            task_id,
            f"{Colors.WARNING}⚠ ERROR: {error_msg}{Colors.ENDC}"
        )

        return {
            "passed": False,
            "iterations": 0,
            "cost_usd": budget_monitor.current_cost,
            "duration_s": 0.0,
            "stop_reason": "error",
            "num_conflicts": 0,
            "num_branches": 0,
            "error": error_msg,
        }


async def main():
    """Main entry point"""

    print_header("GAIA HumanEval Full Benchmark")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{Colors.FAIL}❌ Error: OPENAI_API_KEY environment variable not set!{Colors.ENDC}")
        print("Please set it before running:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1

    # Paths
    data_path = PROJECT_ROOT / "data" / "humaneval" / "test.jsonl"
    checkpoint_path = PROJECT_ROOT / "results" / "humaneval_full_v2.checkpoint.json"
    output_path = PROJECT_ROOT / "results" / "humaneval_full_v2.results.json"
    log_dir = PROJECT_ROOT / "results" / "logs"

    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)

    # Load problems
    print("Loading HumanEval problems...")
    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"✓ Loaded {len(problems)} problems")
    print(f"✓ Per-problem logs → {log_dir}\n")

    # Initialize checkpoint manager
    checkpoint = CheckpointManager(checkpoint_path)
    checkpoint.start_run(total_problems=len(problems))

    # Check if resuming
    completed_task_ids = checkpoint.get_completed_task_ids()
    if completed_task_ids:
        print(f"{Colors.WARNING}Resuming from checkpoint:{Colors.ENDC}")
        print(f"  Already completed: {len(completed_task_ids)} problems")
        print(f"  Resuming at problem #{len(completed_task_ids) + 1}")
        print_stats(checkpoint)

    # Initialize LLMs
    print("Initializing LLMs...")
    fast_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.SLOW)
    print("✓ LLMs initialized\n")

    # Budget monitor
    budget_monitor = BudgetMonitor(max_cost_per_problem=0.30, max_llm_calls=30)

    try:
        for i, problem in enumerate(problems):
            task_id = problem["task_id"]

            # Skip already-completed problems (resume support)
            if checkpoint.is_completed(task_id):
                continue

            # ── Per-problem log file ─────────────────────────────────────────
            # Converts "HumanEval/42" → "HumanEval_42.jsonl"
            task_id_safe = task_id.replace("/", "_")
            log_file = log_dir / f"{task_id_safe}.jsonl"

            # BlackboardLogger: write all events to file, suppress console noise
            bb_logger = BlackboardLogger(log_file=log_file, log_to_console=False)

            # Fresh blackboard + agents for each problem
            blackboard = Blackboard(storage=InMemoryStorage(), logger=bb_logger)

            agents = [
                CoderAgent(llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
                CoderAgent(llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
                CriticAgent(llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
                VerifierAgent(llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
                EdgeCaseAgent(llm=slow_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            ]

            # Run problem
            result = await run_single_problem(
                problem=problem,
                problem_idx=i,
                total_problems=len(problems),
                agents=agents,
                blackboard=blackboard,
                budget_monitor=budget_monitor,
                max_iterations=10,
            )

            # Save checkpoint (atomic write — safe against crashes)
            checkpoint.add_result(
                task_id=task_id,
                passed=result["passed"],
                iterations=result["iterations"],
                cost_usd=result["cost_usd"],
                duration_s=result["duration_s"],
                stop_reason=result["stop_reason"],
                num_conflicts=result["num_conflicts"],
                num_branches=result["num_branches"],
                error=result["error"],
            )

            # Print rolling stats every 10 problems
            if checkpoint.completed_count % 10 == 0:
                print_stats(checkpoint)

        # ── Run complete ─────────────────────────────────────────────────────
        print_header("FINAL RESULTS")
        print_stats(checkpoint)

        final_data = checkpoint.finalize(output_path)
        print(f"{Colors.OKGREEN}✓ Results saved to: {output_path}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ Event logs saved to: {log_dir}/{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ Checkpoint cleaned up{Colors.ENDC}\n")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠ Interrupted by user{Colors.ENDC}")
        print(f"Checkpoint saved → {checkpoint_path}")
        print(f"Event logs saved → {log_dir}")
        print("Run again to resume from where you left off.\n")
        print_stats(checkpoint)
        return 130

    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Fatal error: {e}{Colors.ENDC}")
        traceback.print_exc()
        print(f"\nCheckpoint saved → {checkpoint_path}")
        print_stats(checkpoint)
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted{Colors.ENDC}")
        sys.exit(130)
