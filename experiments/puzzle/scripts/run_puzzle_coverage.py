#!/usr/bin/env python3
"""Run the Information Asymmetry Scaling experiment (E4).

Claim: GAIA degrades gracefully as per-agent clue coverage shrinks. Isolated
agents collapse. The "Specification Gap" 2025 paper identified this as a key
evaluation axis; our robustness curve is the experimental answer.

Mechanism: each expert agent normally receives the full partition (~6 of 12
clues for 4×3 puzzles). At coverage=0.5, each expert receives a random 50% of
their partition's clues, so the total information available across both
partitions is ~6 of 12 clues — still enough for synthesis IF the blackboard
allows agents to share what they've deduced. Isolated agents cannot share, so
they collapse rapidly as coverage drops.

Conditions:
  isolated : 2 independent synthesizers, each sees its own partition. Coverage
             affects how many clues each partition shows.
  gaia     : Full blackboard system. Coverage affects what each expert sees;
             experts post deductions, synthesizers merge everything.

Coverage levels: 0.25 | 0.50 | 0.75 | 1.00.

Folder structure:
  experiments/puzzle/
    scripts/run_puzzle_coverage.py  ← this file (does NOT touch run_puzzle.py)
    results/coverage/               ← JSON + checkpoints
    logs/coverage/                  ← per-puzzle JSONL logs

Usage:
  python experiments/puzzle/scripts/run_puzzle_coverage.py --coverage 0.5
  python experiments/puzzle/scripts/run_puzzle_coverage.py --coverage all
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
from gaia.agents.puzzle import ExpertAgent, SynthesizerAgent, PuzzleCriticAgent, PuzzleVerifierAgent
from gaia.agents.puzzle.synthesizer import parse_solution_from_text
from gaia.agents.puzzle.puzzle_verifier import proposed_matches_ground_truth
from gaia.prompts.puzzle.synthesizer import SynthesizerPrompts
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.puzzle_loop import PuzzleEpisodeLoop, PuzzleEpisodeResult
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "puzzle" / "puzzles.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results" / "coverage"
LOGS_DIR       = EXPERIMENT_DIR / "logs" / "coverage"

FAST_MODEL = "gpt-4.1-mini"
SLOW_MODEL = "gpt-4.1"

COVERAGE_LEVELS = [0.25, 0.50, 0.75, 1.00]


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


def progress(i, n, pid, msg):
    print(f"{C.CYAN}[{i}/{n}]{C.END} {pid:25s} {msg}")


def stats(results, condition, coverage):
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"\n{C.BOLD}── {condition.upper()} @ coverage={coverage:.0%} ──{C.END}")
    print(f"  Completed: {total}  Passed: {C.GREEN}{passed}{C.END}  "
          f"Accuracy: {C.BOLD}{passed/total if total else 0:.1%}{C.END}  "
          f"Cost: ${cost:.4f}\n")


def sample_clues(clues: List[dict], coverage: float, seed: int) -> List[dict]:
    """Sample `coverage * len(clues)` clues deterministically."""
    if coverage >= 1.0:
        return list(clues)
    n_keep = max(1, round(len(clues) * coverage))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(clues)), n_keep))
    return [clues[i] for i in indices]


def make_subsampled_puzzle(puzzle: dict, coverage: float, seed: int) -> dict:
    """Return a new puzzle dict with clues sub-sampled at the given coverage.

    The verifier still uses the FULL all_clues_structs (so it can prove the
    solution is correct). Only what experts see is reduced.
    """
    p = copy.deepcopy(puzzle)
    pid = p["puzzle_id"]
    # Sample independently per partition
    p["clues_a"] = sample_clues(p["clues_a"], coverage,
                                 seed=hash((pid, "a", seed)) & 0xffffffff)
    p["clues_b"] = sample_clues(p["clues_b"], coverage,
                                 seed=hash((pid, "b", seed)) & 0xffffffff)
    # all_clues used by verifier — keep full set so verifier knows what
    # constraints the puzzle should satisfy. The visible clues to agents
    # are only clues_a + clues_b after subsampling.
    return p


# ===========================================================================
# Isolated condition with subsampled clues
# ===========================================================================

async def run_isolated_subsampled(puzzle, idx, total, fast_llm, slow_llm, budget, log_dir, coverage):
    pid = puzzle["puzzle_id"]
    budget.reset()

    ground_truth = puzzle["solution"]
    clues_a_texts = [c["text"] for c in puzzle["clues_a"]]
    clues_b_texts = [c["text"] for c in puzzle["clues_b"]]

    prompts = SynthesizerPrompts()
    safe_id = pid.replace("/", "_")

    deductions_a = [
        ("ExpertA", "A",
         f"Clues from Partition A (coverage={coverage:.0%}):\n"
         + "\n".join(f"  {c}" for c in clues_a_texts)),
    ]
    deductions_b = [
        ("ExpertB", "B",
         f"Clues from Partition B (coverage={coverage:.0%}):\n"
         + "\n".join(f"  {c}" for c in clues_b_texts)),
    ]

    bb_a = Blackboard(log_file=log_dir / f"{safe_id}_isolated_a.jsonl")
    bb_b = Blackboard(log_file=log_dir / f"{safe_id}_isolated_b.jsonl")

    synth_a = SynthesizerAgent(
        name="IsolatedSynth-A",
        tier=ModelTier.SLOW, llm=slow_llm,
        blackboard=bb_a, metrics=MetricsCollector(),
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )
    synth_b = SynthesizerAgent(
        name="IsolatedSynth-B",
        tier=ModelTier.SLOW, llm=slow_llm,
        blackboard=bb_b, metrics=MetricsCollector(),
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )

    async def call_synth(agent, deductions):
        msgs = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.format_user(deductions)},
        ]
        resp = await agent.call_llm(msgs, temperature=0.1)
        return parse_solution_from_text(resp)

    t0 = time.time()
    proposed_a, proposed_b = await asyncio.gather(
        call_synth(synth_a, deductions_a),
        call_synth(synth_b, deductions_b),
    )
    duration_s = time.time() - t0

    passed_a = bool(proposed_a and proposed_matches_ground_truth(proposed_a, ground_truth)[0])
    passed_b = bool(proposed_b and proposed_matches_ground_truth(proposed_b, ground_truth)[0])
    proposed = proposed_a or proposed_b
    passed = passed_a or passed_b
    total_cost = synth_a.budget_monitor.current_cost + synth_b.budget_monitor.current_cost

    return {
        "puzzle_id": pid,
        "condition": "isolated",
        "coverage": coverage,
        "passed": passed,
        "proposed_solution": proposed,
        "cost_usd": total_cost,
        "duration_s": round(duration_s, 2),
        "conflict_detected": False,
        "error": None,
    }


# ===========================================================================
# GAIA condition with subsampled clues
# ===========================================================================

async def run_gaia_subsampled(puzzle, idx, total, fast_llm, slow_llm, budget, log_dir, coverage):
    pid = puzzle["puzzle_id"]
    budget.reset()
    budget.max_cost_per_problem = 0.50

    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_gaia.jsonl")
    metrics = MetricsCollector()
    policy = Policy(max_iterations=20, stop_on_first_pass=True,
                    verification_strictness="all_tests_pass")

    agents = [
        ExpertAgent(name="Expert-A-1", partition="A",
                    tier=ModelTier.FAST, llm=fast_llm,
                    blackboard=bb, metrics=metrics, budget_monitor=budget),
        ExpertAgent(name="Expert-A-2", partition="A",
                    tier=ModelTier.FAST, llm=fast_llm,
                    blackboard=bb, metrics=metrics, budget_monitor=budget),
        ExpertAgent(name="Expert-B-1", partition="B",
                    tier=ModelTier.FAST, llm=fast_llm,
                    blackboard=bb, metrics=metrics, budget_monitor=budget),
        ExpertAgent(name="Expert-B-2", partition="B",
                    tier=ModelTier.FAST, llm=fast_llm,
                    blackboard=bb, metrics=metrics, budget_monitor=budget),
        SynthesizerAgent(name="Synthesizer-1",
                         tier=ModelTier.SLOW, llm=slow_llm,
                         blackboard=bb, metrics=metrics, budget_monitor=budget),
        SynthesizerAgent(name="Synthesizer-2",
                         tier=ModelTier.SLOW, llm=slow_llm,
                         blackboard=bb, metrics=metrics, budget_monitor=budget),
        PuzzleCriticAgent(name="Critic",
                          tier=ModelTier.FAST, llm=fast_llm,
                          blackboard=bb, metrics=metrics, budget_monitor=budget),
        PuzzleVerifierAgent(name="Verifier",
                            tier=ModelTier.FAST, llm=fast_llm,
                            blackboard=bb, metrics=metrics, budget_monitor=budget),
    ]

    loop = PuzzleEpisodeLoop(blackboard=bb, agents=agents,
                              metrics=metrics, policy=policy,
                              budget_monitor=budget)

    t0 = time.time()
    result: PuzzleEpisodeResult = await loop.run_episode(puzzle)
    duration_s = time.time() - t0

    return {
        "puzzle_id": pid,
        "condition": "gaia",
        "coverage": coverage,
        "passed": result.passed,
        "proposed_solution": result.proposed_solution,
        "cost_usd": result.cost_usd,
        "duration_s": duration_s,
        "conflict_detected": result.conflict_detected,
        "conflict_resolved": result.conflict_resolved,
        "num_expert_deductions": result.num_expert_deductions,
        "phase_timings": result.phase_timings,
        "error": result.error,
    }


# ===========================================================================
# Runner
# ===========================================================================

def load_puzzles(n: Optional[int] = None) -> List[dict]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    if n:
        puzzles = puzzles[:n]
    return puzzles


async def run_condition(condition, puzzles, coverage, fast_llm, slow_llm, log_dir, checkpoint, seed):
    results = []
    budget = BudgetMonitor(max_cost_per_problem=0.50, max_iterations=20, max_llm_calls=30)
    n = len(puzzles)
    completed_ids = set(checkpoint.get_completed_task_ids())

    for idx, puzzle in enumerate(puzzles, 1):
        pid = puzzle["puzzle_id"]
        ckpt_key = f"{condition}/c{coverage}/{pid}"

        if ckpt_key in completed_ids:
            progress(idx, n, pid, f"{C.YELLOW}(cached){C.END}")
            continue

        # Subsample clues at the given coverage level
        sub_puzzle = make_subsampled_puzzle(puzzle, coverage, seed=seed)

        try:
            if condition == "isolated":
                r = await run_isolated_subsampled(sub_puzzle, idx, n, fast_llm, slow_llm, budget, log_dir, coverage)
            elif condition == "gaia":
                r = await run_gaia_subsampled(sub_puzzle, idx, n, fast_llm, slow_llm, budget, log_dir, coverage)
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

            status = f"{C.GREEN}PASS{C.END}" if r["passed"] else f"{C.FAIL}FAIL{C.END}"
            progress(idx, n, pid, f"{status}  cost=${r.get('cost_usd', 0):.4f}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {"puzzle_id": pid, "condition": condition, "coverage": coverage,
                   "passed": False, "error": str(e), "cost_usd": 0}
            results.append(err)
            checkpoint.add_result(
                task_id=ckpt_key, passed=False, iterations=1,
                cost_usd=0, duration_s=0, stop_reason="error", error=str(e),
            )
            print(f"  {C.FAIL}ERROR{C.END} {pid}: {e}")
            traceback.print_exc()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="E4 Coverage Scaling")
    parser.add_argument(
        "--coverage", type=str, default="all",
        help="Coverage level (0.25, 0.5, 0.75, 1.0) or 'all'",
    )
    parser.add_argument("--condition", choices=["isolated", "gaia", "all"], default="all")
    parser.add_argument("--n_puzzles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    puzzles = load_puzzles(args.n_puzzles)

    fast_llm = OpenAILLM(model=FAST_MODEL, tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model=SLOW_MODEL, tier=ModelTier.SLOW)

    if args.coverage == "all":
        coverages = COVERAGE_LEVELS
    else:
        coverages = [float(args.coverage)]

    if args.condition == "all":
        conditions = ["isolated", "gaia"]
    else:
        conditions = [args.condition]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for coverage in coverages:
        for condition in conditions:
            hdr(f"E4 Coverage — {condition.upper()} @ {coverage:.0%}  ({len(puzzles)} puzzles)")

            log_dir = LOGS_DIR / f"{timestamp}_{condition}_c{int(coverage*100)}"
            log_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = RESULTS_DIR / f"checkpoint_{condition}_c{int(coverage*100)}_{timestamp}.json"
            checkpoint = CheckpointManager(ckpt_path)

            results = await run_condition(
                condition, puzzles, coverage, fast_llm, slow_llm,
                log_dir, checkpoint, args.seed,
            )

            stats(results, condition, coverage)
            key = f"{condition}_c{int(coverage*100)}"
            all_results[key] = {
                "results": results,
                "summary": {
                    "condition": condition,
                    "coverage": coverage,
                    "n_puzzles": len(results),
                    "n_passed": sum(1 for r in results if r.get("passed")),
                    "accuracy": sum(1 for r in results if r.get("passed")) / len(results) if results else 0,
                    "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
                },
            }

    out_path = RESULTS_DIR / f"coverage_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{C.BOLD}Results saved to: {out_path}{C.END}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
