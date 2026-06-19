#!/usr/bin/env python3
"""Run the Correlated Failure experiment (E3).

Claim: Majority vote fails when 2/3 agents share the same systematic reasoning
error (a "trap"). GAIA's reconciler — which receives all three reasoning chains
alongside the conflict signal — identifies the systematic error and corrects it.

Experimental design:
  Agent pool (GAIA condition):
    2 × MathSolverAgent  (standard, prone to traps at temperature 0.0 and 0.3)
    1 × TrapAwareSolverAgent  (explicitly self-audits for the 5 trap categories)
    + MathAggregator + MathReconciler + MathVerifier

  On a trap problem, the 2 standard solvers BOTH fall in (same wrong answer).
  The trap-aware solver gets it right. That creates a 2-vs-1 majority that
  is WRONG. The reconciler must override this majority.

Conditions:
  single        : 1 standard MathSolverAgent (no consensus check)
  majority_vote : 2 standard + 1 trap-aware, majority wins (no reconciler)
  gaia          : Full pool — reconciler can override the majority

Folder structure:
  experiments/correlated_failure/
    scripts/run_correlated_failure.py  ← this file
    results/                            ← JSON + checkpoints
    logs/                               ← per-problem JSONL logs

Usage:
  python experiments/correlated_failure/scripts/run_correlated_failure.py
  python experiments/correlated_failure/scripts/run_correlated_failure.py --condition gaia
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-load .env
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy
from gaia.agents.math import (
    MathSolverAgent, MathAggregatorAgent,
    MathReconcilerAgent, MathVerifierAgent,
)
from gaia.agents.math.math_solver import extract_final_answer
from gaia.agents.math.misled_solver import MisledSolverAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.correlated_failure_loop import (
    CorrelatedFailureEpisodeLoop, CorrelatedFailureEpisodeResult,
)
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "gsm8k" / "correlated_failure_problems.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results"
LOGS_DIR       = EXPERIMENT_DIR / "logs"

FAST_MODEL = "gpt-4.1-nano"
SLOW_MODEL = "gpt-4.1"


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
    print(f"{C.CYAN}[{i}/{n}]{C.END} {pid:30s} {msg}")


def stats(results: List[dict], condition: str):
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    cost = sum(r.get("cost_usd", 0) for r in results)
    rate = passed / total if total else 0
    print(f"\n{C.BOLD}── {condition.upper()} Statistics ──{C.END}")
    print(f"  Completed : {total}")
    print(f"  Passed    : {C.GREEN}{passed}{C.END}")
    print(f"  Failed    : {C.FAIL}{total - passed}{C.END}")
    print(f"  Accuracy  : {C.BOLD}{rate:.1%}{C.END}")
    print(f"  Cost      : ${cost:.4f}\n")


# =============================================================================
# Condition 1: Single standard solver
# =============================================================================

async def run_single(problem, idx, total, fast_llm, budget, log_dir):
    from gaia.prompts.math.solver import MathSolverPrompts

    pid = problem["problem_id"]
    budget.reset()
    ground_truth = problem["answer"]
    question = problem["question"]

    prompts = MathSolverPrompts()
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_single.jsonl")
    metrics = MetricsCollector()

    agent = MathSolverAgent(
        solver_index=0, name="SingleSolver",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb, metrics=metrics, budget_monitor=budget,
    )

    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user", "content": prompts.format_user(question)},
    ]

    t0 = time.time()
    response = await agent.call_llm(messages, temperature=0.0)
    duration_s = time.time() - t0

    proposed_answer = extract_final_answer(response)
    passed = (proposed_answer is not None and proposed_answer == ground_truth)

    r = {
        "problem_id": pid, "condition": "single",
        "passed": passed,
        "proposed_answer": proposed_answer,
        "ground_truth": ground_truth,
        "common_wrong_answer": problem.get("common_wrong_answer"),
        "trap_category": problem.get("category"),
        "cost_usd": budget.current_cost,
        "duration_s": round(duration_s, 2),
        "error": None,
    }

    status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.FAIL}FAIL{C.END}"
    progress(idx, total, pid, f"{status}  ans={proposed_answer}  truth={ground_truth}")
    return r


# =============================================================================
# Condition 2: Majority vote (2 misled + 1 clean, NO reconciler)
# Same agent pool as GAIA but no reconciler — the wrong 2/3 majority wins.
# =============================================================================

async def run_majority_vote(problem, idx, total, fast_llm, budget, log_dir):
    from gaia.prompts.math.solver import MathSolverPrompts
    from gaia.prompts.math.misled_solver import MisledSolverPrompts

    pid = problem["problem_id"]
    budget.reset()
    ground_truth = problem["answer"]
    question = problem["question"]
    hint = problem.get("misleading_hint", "")

    safe_id = pid.replace("/", "_")
    clean_prompts = MathSolverPrompts()
    misled_prompts = MisledSolverPrompts()

    bb0 = Blackboard(log_file=log_dir / f"{safe_id}_majority_misled0.jsonl")
    bb1 = Blackboard(log_file=log_dir / f"{safe_id}_majority_misled1.jsonl")
    bb2 = Blackboard(log_file=log_dir / f"{safe_id}_majority_clean.jsonl")

    misled0 = MisledSolverAgent(
        misled_index=0, name="MisledSolver-0",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb0, metrics=MetricsCollector(),
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )
    misled1 = MisledSolverAgent(
        misled_index=1, name="MisledSolver-1",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb1, metrics=MetricsCollector(),
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )
    clean = MathSolverAgent(
        solver_index=0, name="CleanSolver",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb2, metrics=MetricsCollector(),
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )

    async def call_misled(agent):
        msgs = [
            {"role": "system", "content": misled_prompts.SYSTEM},
            {"role": "user", "content": misled_prompts.format_user(question, hint)},
        ]
        return extract_final_answer(await agent.call_llm(msgs, temperature=0.0))

    async def call_clean(agent):
        msgs = [
            {"role": "system", "content": clean_prompts.SYSTEM},
            {"role": "user", "content": clean_prompts.format_user(question)},
        ]
        return extract_final_answer(await agent.call_llm(msgs, temperature=0.0))

    t0 = time.time()
    m0, m1, c = await asyncio.gather(
        call_misled(misled0), call_misled(misled1), call_clean(clean),
    )
    duration_s = time.time() - t0

    all_answers = [m0, m1, c]
    valid = [a for a in all_answers if a is not None]
    proposed_answer = Counter(valid).most_common(1)[0][0] if valid else None
    passed = (proposed_answer is not None and proposed_answer == ground_truth)
    total_cost = (misled0.budget_monitor.current_cost
                  + misled1.budget_monitor.current_cost
                  + clean.budget_monitor.current_cost)

    misled_vals = [v for v in (m0, m1) if v is not None]
    correlated_failure = (
        len(misled_vals) == 2 and len(set(misled_vals)) == 1
        and misled_vals[0] != ground_truth
    )

    r = {
        "problem_id": pid, "condition": "majority_vote",
        "passed": passed,
        "proposed_answer": proposed_answer,
        "ground_truth": ground_truth,
        "common_wrong_answer": problem.get("common_wrong_answer"),
        "trap_category": problem.get("category"),
        "misled_answers": {"Misled-0": m0, "Misled-1": m1},
        "clean_answer": c,
        "majority_answer": proposed_answer,
        "correlated_failure_present": correlated_failure,
        "conflict_detected": len(set(valid)) > 1 if valid else False,
        "cost_usd": total_cost,
        "duration_s": round(duration_s, 2),
        "error": None,
    }

    status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.FAIL}FAIL{C.END}"
    cf = f" {C.YELLOW}[misled={m0},{m1} clean={c}]{C.END}" if correlated_failure else ""
    progress(idx, total, pid, f"{status}  ans={proposed_answer}  truth={ground_truth}{cf}")
    return r


# =============================================================================
# Condition 3: Full GAIA system (2 misled + 1 clean + reconciler)
# =============================================================================

async def run_gaia(problem, idx, total, fast_llm, slow_llm, budget, log_dir):
    pid = problem["problem_id"]
    budget.reset()
    safe_id = pid.replace("/", "_")

    bb = Blackboard(log_file=log_dir / f"{safe_id}_gaia.jsonl")
    metrics = MetricsCollector()
    policy = Policy()

    agents = [
        MisledSolverAgent(
            misled_index=0, name="MisledSolver-0",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MisledSolverAgent(
            misled_index=1, name="MisledSolver-1",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathSolverAgent(
            solver_index=0, name="CleanSolver",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathAggregatorAgent(
            name="MathAggregator",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathReconcilerAgent(
            name="MathReconciler",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        MathVerifierAgent(
            name="MathVerifier",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
    ]

    loop = CorrelatedFailureEpisodeLoop(
        blackboard=bb, agents=agents,
        metrics=metrics, policy=policy,
        budget_monitor=budget,
    )

    try:
        result: CorrelatedFailureEpisodeResult = await loop.run_episode(problem)
        r = {
            "problem_id": pid, "condition": "gaia",
            "passed": result.passed,
            "proposed_answer": result.proposed_answer,
            "ground_truth": result.ground_truth,
            "common_wrong_answer": problem.get("common_wrong_answer"),
            "trap_category": problem.get("category"),
            "misled_answers": result.misled_answers,
            "clean_answer": result.clean_answer,
            "majority_answer": result.majority_answer,
            "correlated_failure_present": result.correlated_failure_present,
            "conflict_detected": result.conflict_detected,
            "conflict_resolved": result.conflict_resolved,
            "reconciler_sided_with_clean": result.reconciler_sided_with_clean,
            "cost_usd": result.cost_usd,
            "duration_s": result.duration_s,
            "phase_timings": result.phase_timings,
            "stop_reason": result.stop_reason,
            "error": result.error,
        }
    except Exception as e:
        r = {
            "problem_id": pid, "condition": "gaia",
            "passed": False, "error": str(e),
            "cost_usd": budget.current_cost,
        }
        traceback.print_exc()

    status = f"{C.GREEN}PASS{C.END}" if r.get("passed") else f"{C.FAIL}FAIL{C.END}"
    conflict_flag = (
        f" {C.YELLOW}[CONFLICT resolved]{C.END}"
        if r.get("conflict_resolved") else ""
    )
    progress(idx, total, pid, f"{status}  ans={r.get('proposed_answer')}  truth={r.get('ground_truth')}{conflict_flag}")
    return r


# =============================================================================
# Shared runner with checkpointing
# =============================================================================

def load_problems(n: Optional[int] = None) -> List[dict]:
    with open(DATA_PATH) as f:
        problems = json.load(f)
    if n:
        problems = problems[:n]
    return problems


async def run_condition(condition, problems, fast_llm, slow_llm, log_dir, checkpoint):
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


def parse_args():
    parser = argparse.ArgumentParser(description="E3 Correlated Failure Experiment")
    parser.add_argument(
        "--condition",
        choices=["single", "majority_vote", "gaia", "all"],
        default="all",
    )
    parser.add_argument("--n_problems", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast_model", type=str, default=None,
                        help="override FAST tier model (NX3 cross-model)")
    parser.add_argument("--slow_model", type=str, default=None,
                        help="override SLOW tier model (NX3 cross-model)")
    return parser.parse_args()


async def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    problems = load_problems(args.n_problems)

    fast_llm = OpenAILLM(model=args.fast_model or FAST_MODEL, tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model=args.slow_model or SLOW_MODEL, tier=ModelTier.SLOW)
    if args.fast_model or args.slow_model:
        print(f"[NX3 cross-model] fast={args.fast_model or FAST_MODEL} "
              f"slow={args.slow_model or SLOW_MODEL}")

    conditions = (
        ["single", "majority_vote", "gaia"]
        if args.condition == "all" else [args.condition]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for condition in conditions:
        hdr(f"E3 Correlated Failure — {condition.upper().replace('_', ' ')} ({len(problems)} problems)")

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
                "accuracy": (
                    sum(1 for r in results if r.get("passed")) / len(results)
                    if results else 0
                ),
                "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
                "n_conflicts": sum(1 for r in results if r.get("conflict_detected")),
            },
        }

    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / f"correlated_failure_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{C.BOLD}Results saved to: {out_path}{C.END}")

    if len(all_results) > 1:
        print(f"\n{C.BOLD}── Comparison Summary ──{C.END}")
        print(f"{'Condition':<18} {'Accuracy':<12} {'Cost'}")
        print("─" * 50)
        for cond, data in all_results.items():
            s = data["summary"]
            acc = f"{s['accuracy']:.1%}"
            cost = f"${s['total_cost_usd']:.4f}"
            print(f"{cond:<18} {acc:<12} {cost}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
