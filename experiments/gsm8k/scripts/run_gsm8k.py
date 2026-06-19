#!/usr/bin/env python3
"""Run the GSM8K Mathematical Reasoning experiment.

Three experimental conditions (for paper ablation):
  --condition single       : 1 solver (temp=0.0), no aggregation — pure single-agent baseline
  --condition majority_vote: 3 solvers, majority-vote answer, NO reconciliation (ablation)
  --condition gaia         : 3 solvers + aggregator + reconciler (full GAIA system)
  --condition all          : run all three in sequence (default)

Models:
  Fast tier (Solvers, Aggregator, Verifier): gpt-4.1-nano
  Slow tier (Reconciler): gpt-4.1

Folder structure:
  experiments/gsm8k/
    scripts/run_gsm8k.py       ← this file
    results/                   ← JSON results + checkpoints
    logs/                      ← per-problem JSONL logs

Usage:
  python experiments/gsm8k/scripts/run_gsm8k.py
  python experiments/gsm8k/scripts/run_gsm8k.py --condition gaia --n_problems 10
  python experiments/gsm8k/scripts/run_gsm8k.py --condition single
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Project root (scripts/ → gsm8k/ → experiments/ → GAIA root) ──────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Auto-load .env ────────────────────────────────────────────────────────────
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy, ArtifactType
from gaia.agents.math import (
    MathSolverAgent, MathAggregatorAgent,
    MathReconcilerAgent, MathVerifierAgent,
)
from gaia.agents.math.math_solver import extract_final_answer
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.math_loop import MathEpisodeLoop, MathEpisodeResult
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "gsm8k" / "extreme_v2_problems.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results"
LOGS_DIR       = EXPERIMENT_DIR / "logs"

# ── Models ────────────────────────────────────────────────────────────────────
FAST_MODEL = "gpt-4.1-nano"   # Solvers, Aggregator, Verifier (weaker → produces conflicts)
SLOW_MODEL = "gpt-4.1"        # Reconciler (hardest reasoning task)


# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    FAIL   = "\033[91m"
    BOLD   = "\033[1m"
    END    = "\033[0m"


def hdr(text: str):
    print(f"\n{C.BOLD}{C.HEADER}{'='*80}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{text.center(80)}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'='*80}{C.END}\n")


def progress(i: int, n: int, pid: str, msg: str):
    print(f"{C.CYAN}[{i}/{n}]{C.END} {pid:25s} {msg}")


def stats(results: List[dict], condition: str):
    passed   = sum(1 for r in results if r.get("passed"))
    total    = len(results)
    cost     = sum(r.get("cost_usd", 0) for r in results)
    conflicts = sum(1 for r in results if r.get("conflict_detected"))
    rate     = passed / total if total else 0
    print(f"\n{C.BOLD}── {condition.upper()} Statistics ──{C.END}")
    print(f"  Completed : {total}")
    print(f"  Passed    : {C.GREEN}{passed}{C.END}")
    print(f"  Failed    : {C.FAIL}{total - passed}{C.END}")
    print(f"  Accuracy  : {C.BOLD}{rate:.1%}{C.END}")
    if condition == "gaia":
        print(f"  Conflicts : {C.YELLOW}{conflicts}{C.END} ({conflicts/total:.1%} of problems)")
    print(f"  Cost      : ${cost:.4f}\n")


# =============================================================================
# Condition 1: Single-agent baseline
# One solver (temperature=0), parse answer directly, verify.
# =============================================================================

async def run_single(
    problem: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
) -> dict:
    """Single-agent baseline: 1 solver, no consensus checking, no reconciliation."""
    from gaia.prompts.math.solver import MathSolverPrompts
    from gaia.utils.metrics import MetricsCollector

    pid = problem["problem_id"]
    budget.reset()
    ground_truth = problem["answer"]
    question = problem["question"]

    prompts = MathSolverPrompts()
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_single.jsonl")
    metrics = MetricsCollector()

    agent = MathSolverAgent(
        solver_index=0,
        name="SingleSolver",
        tier=ModelTier.FAST,
        llm=fast_llm,
        blackboard=bb,
        metrics=metrics,
        budget_monitor=budget,
    )

    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user",   "content": prompts.format_user(question)},
    ]

    t0 = time.time()
    response = await agent.call_llm(messages, temperature=0.0)
    duration_s = time.time() - t0

    proposed_answer = extract_final_answer(response)
    passed = (proposed_answer is not None and proposed_answer == ground_truth)

    result = {
        "problem_id": pid,
        "condition": "single",
        "passed": passed,
        "proposed_answer": proposed_answer,
        "ground_truth": ground_truth,
        "cost_usd": budget.current_cost,
        "duration_s": round(duration_s, 2),
        "conflict_detected": False,
        "error": None,
    }

    status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.FAIL}FAIL{C.END}"
    progress(idx, total, pid, f"{status}  proposed={proposed_answer}  truth={ground_truth}")
    return result


# =============================================================================
# Condition 2: Majority vote (3 solvers, no reconciliation)
# Shows benefit of parallel voting WITHOUT the conflict-resolution mechanism.
# =============================================================================

async def run_majority_vote(
    problem: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
) -> dict:
    """3 parallel solvers, answer = majority vote. No reconciliation step."""
    from gaia.prompts.math.solver import MathSolverPrompts
    from gaia.utils.metrics import MetricsCollector

    pid = problem["problem_id"]
    budget.reset()
    ground_truth = problem["answer"]
    question = problem["question"]

    prompts = MathSolverPrompts()
    safe_id = pid.replace("/", "_")
    metrics = MetricsCollector()

    # Create 3 independent solvers, each with its own blackboard + budget
    temperatures = [0.0, 0.3, 0.6]
    solvers = []
    for i, temp in enumerate(temperatures):
        bb_i = Blackboard(log_file=log_dir / f"{safe_id}_majority_solver{i}.jsonl")
        solver = MathSolverAgent(
            solver_index=i,
            name=f"MajoritySolver-{i+1}",
            tier=ModelTier.FAST,
            llm=fast_llm,
            blackboard=bb_i,
            metrics=MetricsCollector(),
            budget_monitor=BudgetMonitor(
                max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5
            ),
        )
        solvers.append((solver, temp))

    async def call_solver(solver, temp):
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user",   "content": prompts.format_user(question)},
        ]
        response = await solver.call_llm(messages, temperature=temp)
        return extract_final_answer(response)

    t0 = time.time()
    answers = await asyncio.gather(*[call_solver(s, t) for s, t in solvers])
    duration_s = time.time() - t0

    # Majority vote: find the most common non-None answer
    valid = [a for a in answers if a is not None]
    if valid:
        most_common, count = Counter(valid).most_common(1)[0]
        proposed_answer = most_common
    else:
        proposed_answer = None

    passed = (proposed_answer is not None and proposed_answer == ground_truth)
    total_cost = sum(s.budget_monitor.current_cost for s, _ in solvers)

    result = {
        "problem_id": pid,
        "condition": "majority_vote",
        "passed": passed,
        "proposed_answer": proposed_answer,
        "ground_truth": ground_truth,
        "solver_answers": list(answers),
        "cost_usd": total_cost,
        "duration_s": round(duration_s, 2),
        "conflict_detected": len(set(valid)) > 1 if valid else False,
        "error": None,
    }

    status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.FAIL}FAIL{C.END}"
    conflict_flag = f" {C.YELLOW}[conflict: {answers}]{C.END}" if len(set(valid)) > 1 else ""
    progress(idx, total, pid, f"{status}  proposed={proposed_answer}  truth={ground_truth}{conflict_flag}")
    return result


# =============================================================================
# Condition 3: Full GAIA system
# 3 solvers + aggregator + (conditional) reconciler + verifier
# =============================================================================

async def run_gaia(
    problem: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
) -> dict:
    """Full GAIA system: 3 solvers, consensus check, conflict reconciliation."""
    pid = problem["problem_id"]
    budget.reset()
    safe_id = pid.replace("/", "_")

    bb = Blackboard(log_file=log_dir / f"{safe_id}_gaia.jsonl")
    metrics = MetricsCollector()
    policy = Policy()

    # ── Agent roster ──────────────────────────────────────────────────────────
    agents = [
        # 3 parallel solvers (fast tier, different temperatures)
        MathSolverAgent(
            solver_index=0, name="MathSolver-1",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathSolverAgent(
            solver_index=1, name="MathSolver-2",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathSolverAgent(
            solver_index=2, name="MathSolver-3",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Aggregator: checks consensus (fast tier — only extracts integers)
        MathAggregatorAgent(
            name="MathAggregator",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Reconciler: resolves conflicts (slow tier — audits full reasoning chains)
        MathReconcilerAgent(
            name="MathReconciler",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Verifier: integer comparison (fast tier)
        MathVerifierAgent(
            name="MathVerifier",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
    ]

    loop = MathEpisodeLoop(
        blackboard=bb,
        agents=agents,
        metrics=metrics,
        policy=policy,
        budget_monitor=budget,
    )

    try:
        result: MathEpisodeResult = await loop.run_episode(problem)
        r = {
            "problem_id": pid,
            "condition": "gaia",
            "passed": result.passed,
            "proposed_answer": result.proposed_answer,
            "ground_truth": result.ground_truth,
            "solver_answers": result.solver_answers,
            "conflict_detected": result.conflict_detected,
            "conflict_resolved": result.conflict_resolved,
            "cost_usd": result.cost_usd,
            "duration_s": result.duration_s,
            "phase_timings": result.phase_timings,
            "stop_reason": result.stop_reason,
            "error": result.error,
        }
    except Exception as e:
        r = {
            "problem_id": pid,
            "condition": "gaia",
            "passed": False,
            "error": str(e),
            "cost_usd": budget.current_cost,
        }
        traceback.print_exc()

    status = f"{C.GREEN}PASS{C.END}" if r.get("passed") else f"{C.FAIL}FAIL{C.END}"
    conflict_flag = f" {C.YELLOW}[CONFLICT resolved]{C.END}" if r.get("conflict_resolved") else ""
    proposed = r.get("proposed_answer")
    truth = problem["answer"]
    progress(idx, total, pid, f"{status}  proposed={proposed}  truth={truth}{conflict_flag}")
    return r


# =============================================================================
# Shared runner: load data, checkpoint, run, save
# =============================================================================

def load_problems(n: Optional[int] = None) -> List[dict]:
    with open(DATA_PATH) as f:
        problems = json.load(f)
    if n:
        problems = problems[:n]
    return problems


async def run_condition(
    condition: str,
    problems: List[dict],
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    log_dir: Path,
    checkpoint: CheckpointManager,
) -> List[dict]:
    """Run all problems for a given condition with checkpointing."""
    results = []
    budget = BudgetMonitor(max_cost_per_problem=1.00, max_iterations=20, max_llm_calls=30)
    n = len(problems)
    completed_ids = set(checkpoint.get_completed_task_ids())

    for idx, problem in enumerate(problems, 1):
        pid = problem["problem_id"]
        ckpt_key = f"{condition}/{pid}"

        if ckpt_key in completed_ids:
            progress(idx, n, pid, f"{C.YELLOW}(skipped — cached){C.END}")
            continue

        try:
            if condition == "single":
                r = await run_single(problem, idx, n, fast_llm, budget, log_dir)
            elif condition == "majority_vote":
                r = await run_majority_vote(problem, idx, n, fast_llm, budget, log_dir)
            elif condition == "gaia":
                r = await run_gaia(problem, idx, n, fast_llm, slow_llm, budget, log_dir)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            results.append(r)
            checkpoint.add_result(
                task_id=ckpt_key,
                passed=r.get("passed", False),
                iterations=1,
                cost_usd=r.get("cost_usd", 0),
                duration_s=r.get("duration_s", 0),
                stop_reason="passed" if r.get("passed") else "failed",
                num_conflicts=1 if r.get("conflict_detected") else 0,
                error=r.get("error"),
            )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {
                "problem_id": pid, "condition": condition,
                "passed": False, "error": str(e), "cost_usd": 0,
            }
            results.append(err)
            checkpoint.add_result(
                task_id=ckpt_key, passed=False, iterations=1,
                cost_usd=0, duration_s=0, stop_reason="error", error=str(e),
            )
            print(f"  {C.FAIL}ERROR{C.END} {pid}: {e}")
            traceback.print_exc()

    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K Math Reasoning Experiment")
    parser.add_argument(
        "--condition",
        choices=["single", "majority_vote", "gaia", "all"],
        default="all",
    )
    parser.add_argument("--n_problems", type=int, default=None,
                        help="Limit to first N problems (default: all 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Ensure output dirs exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    problems = load_problems(args.n_problems)

    # LLM providers
    fast_llm = OpenAILLM(model=FAST_MODEL, tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model=SLOW_MODEL, tier=ModelTier.SLOW)

    conditions = (
        ["single", "majority_vote", "gaia"]
        if args.condition == "all"
        else [args.condition]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for condition in conditions:
        hdr(f"GSM8K — {condition.upper().replace('_', ' ')} ({len(problems)} problems)")

        log_dir = LOGS_DIR / f"{timestamp}_{condition}"
        log_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = RESULTS_DIR / f"checkpoint_{condition}_{timestamp}.json"
        checkpoint = CheckpointManager(ckpt_path)

        results = await run_condition(
            condition, problems, fast_llm, slow_llm, log_dir, checkpoint
        )

        stats(results, condition)
        all_results[condition] = {
            "results": results,
            "summary": {
                "condition": condition,
                "n_problems": len(results),
                "n_passed": sum(1 for r in results if r.get("passed")),
                "accuracy": sum(1 for r in results if r.get("passed")) / len(results) if results else 0,
                "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
                "n_conflicts": sum(1 for r in results if r.get("conflict_detected")),
            },
        }

    # Save combined results
    out_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"gsm8k_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{C.BOLD}Results saved to: {out_path}{C.END}")

    # Print comparison summary if multiple conditions ran
    if len(all_results) > 1:
        print(f"\n{C.BOLD}── Comparison Summary ──{C.END}")
        print(f"{'Condition':<18} {'Accuracy':<12} {'Conflicts':<12} {'Cost'}")
        print("─" * 55)
        for cond, data in all_results.items():
            s = data["summary"]
            acc = f"{s['accuracy']:.1%}"
            conf = f"{s['n_conflicts']}/{s['n_problems']}"
            cost = f"${s['total_cost_usd']:.4f}"
            print(f"{cond:<18} {acc:<12} {conf:<12} {cost}")


if __name__ == "__main__":
    # Suppress verbose agent/budget INFO logs; keep WARNING+ for unexpected issues
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("math_loop").setLevel(logging.WARNING)
    logging.getLogger("agent.math_solver").setLevel(logging.WARNING)
    logging.getLogger("agent.math_aggregator").setLevel(logging.WARNING)
    logging.getLogger("agent.math_reconciler").setLevel(logging.WARNING)
    logging.getLogger("agent.math_verifier").setLevel(logging.WARNING)
    logging.getLogger("budget").setLevel(logging.WARNING)
    asyncio.run(main())
