#!/usr/bin/env python3
"""Run the Agent Scaling experiment (E8).

Claim: Adding agents to GAIA improves accuracy (up to a point). Adding agents
WITHOUT coordination (homogeneous baseline) provides no benefit regardless of
count. This separates three effects: count, coordination, and specialization.

Conditions:
  homogeneous_N : N identical GeneralistAgents that each see ALL clues,
                  independently solve, majority-vote the answer. NO blackboard.
  gaia_N        : Role-specialized agents with full blackboard coordination.
                  For N=4: 1 ExpertA + 1 ExpertB + 1 Synthesizer + 1 Critic
                  For N=6: 2 ExpertA + 2 ExpertB + 1 Synthesizer + 1 Critic
                  For N=8: 2 ExpertA + 2 ExpertB + 2 Synthesizer + 1 Critic + 1 Verifier
                  (Verifier is always present for ground-truth check.)

Folder structure:
  experiments/puzzle/
    scripts/run_puzzle_scaling_agents.py  ← this file
    results/scaling/                       ← JSON + checkpoints
    logs/scaling/                          ← per-puzzle JSONL logs

Usage:
  python experiments/puzzle/scripts/run_puzzle_scaling_agents.py --num-agents 4 --condition-type homogeneous
  python experiments/puzzle/scripts/run_puzzle_scaling_agents.py --num-agents 4 --condition-type gaia
  python experiments/puzzle/scripts/run_puzzle_scaling_agents.py --num-agents all --condition-type all
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
from gaia.agents.puzzle.generalist_agent import GeneralistAgent
from gaia.agents.puzzle.synthesizer import parse_solution_from_text
from gaia.agents.puzzle.puzzle_verifier import proposed_matches_ground_truth
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.puzzle_loop import PuzzleEpisodeLoop, PuzzleEpisodeResult
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "puzzle" / "puzzles.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results" / "scaling"
LOGS_DIR       = EXPERIMENT_DIR / "logs" / "scaling"

FAST_MODEL = "gpt-4.1-mini"
SLOW_MODEL = "gpt-4.1"

DEFAULT_PEOPLE = ["Alice", "Bob", "Carol", "Dave"]
DEFAULT_ATTRIBUTES = {
    "job": ["doctor", "artist", "teacher", "engineer"],
    "pet": ["dog", "cat", "fish", "bird"],
    "drink": ["water", "coffee", "juice", "tea"],
}


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


# ===========================================================================
# Homogeneous baseline (N GeneralistAgents, no coordination, majority vote)
# ===========================================================================

async def run_homogeneous(puzzle, idx, total, fast_llm, num_agents, log_dir):
    """N identical generalists solve in parallel, no sharing, majority vote."""
    pid = puzzle["puzzle_id"]
    safe_id = pid.replace("/", "_")
    ground_truth = puzzle["solution"]
    all_clues_texts = [c["text"] for c in puzzle["all_clues"]]

    # Each generalist runs independently on its own blackboard
    generalists = []
    for i in range(num_agents):
        bb_i = Blackboard(log_file=log_dir / f"{safe_id}_homo{num_agents}_g{i}.jsonl")
        # Use a wider range of temperatures to encourage diversity
        agent = GeneralistAgent(
            generalist_index=i,
            people=DEFAULT_PEOPLE,
            attributes=DEFAULT_ATTRIBUTES,
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb_i, metrics=MetricsCollector(),
            budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
        )
        generalists.append(agent)

    # Diverse temperatures
    temps = [0.0, 0.4, 0.2, 0.6, 0.1, 0.5][:num_agents]

    from gaia.prompts.puzzle.generalist import GeneralistPrompts
    prompts = GeneralistPrompts()

    async def call_generalist(agent, temp):
        msgs = [
            {"role": "system", "content": prompts.format_system(len(DEFAULT_PEOPLE))},
            {"role": "user", "content": prompts.format_user(
                clues=all_clues_texts,
                people=DEFAULT_PEOPLE,
                attributes=DEFAULT_ATTRIBUTES,
            )},
        ]
        resp = await agent.call_llm(msgs, temperature=temp)
        return parse_solution_from_text(resp)

    t0 = time.time()
    solutions = await asyncio.gather(*[call_generalist(g, t) for g, t in zip(generalists, temps)])
    duration_s = time.time() - t0

    # Majority vote on the parsed solutions
    # Compare solutions as tuples of (person, attr, val) for hashable equality
    def solution_key(sol):
        if sol is None:
            return None
        return tuple(sorted(
            (p, a, v) for p, attrs in sol.items() for a, v in attrs.items()
        ))

    keys = [solution_key(s) for s in solutions]
    valid_keys = [k for k in keys if k is not None]

    proposed = None
    if valid_keys:
        most_common_key, count = Counter(valid_keys).most_common(1)[0]
        # Pick the actual solution that matches the most common key
        for sol, key in zip(solutions, keys):
            if key == most_common_key:
                proposed = sol
                break

    passed = False
    if proposed:
        passed, _ = proposed_matches_ground_truth(proposed, ground_truth)

    total_cost = sum(g.budget_monitor.current_cost for g in generalists)

    r = {
        "puzzle_id": pid,
        "condition": f"homogeneous_{num_agents}",
        "num_agents": num_agents,
        "condition_type": "homogeneous",
        "passed": passed,
        "proposed_solution": proposed,
        "n_unique_solutions": len(set(valid_keys)),
        "cost_usd": total_cost,
        "duration_s": round(duration_s, 2),
        "conflict_detected": len(set(valid_keys)) > 1,
        "error": None,
    }

    status = f"{C.GREEN}PASS{C.END}" if passed else f"{C.FAIL}FAIL{C.END}"
    progress(idx, total, pid,
             f"{status}  n={num_agents}  unique={len(set(valid_keys))}  cost=${total_cost:.4f}")
    return r


# ===========================================================================
# GAIA with N agents (role-specialized, full blackboard)
# ===========================================================================

def build_gaia_agents(num_agents, bb, fast_llm, slow_llm, metrics, budget):
    """Build a role-balanced agent pool for the GAIA condition.

    Always includes: 1 Critic, 1 Verifier. The remaining (num_agents - 2)
    slots are split among Experts (split A/B) and Synthesizers in a fixed
    proportion: roughly 2/3 experts, 1/3 synthesizers, with at least 1 of each.
    """
    n_remaining = max(2, num_agents - 2)  # at least 2 for expert+synth
    # Allocate: aim for ~75% experts, ~25% synth
    n_synth = max(1, n_remaining // 4)
    n_experts = max(2, n_remaining - n_synth)
    # Split experts evenly between A and B
    n_a = n_experts // 2 + (n_experts % 2)
    n_b = n_experts // 2

    agents = []
    for i in range(n_a):
        agents.append(ExpertAgent(
            name=f"Expert-A-{i+1}", partition="A",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ))
    for i in range(n_b):
        agents.append(ExpertAgent(
            name=f"Expert-B-{i+1}", partition="B",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ))
    for i in range(n_synth):
        agents.append(SynthesizerAgent(
            name=f"Synthesizer-{i+1}",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ))
    agents.append(PuzzleCriticAgent(
        name="Critic",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb, metrics=metrics, budget_monitor=budget,
    ))
    agents.append(PuzzleVerifierAgent(
        name="Verifier",
        tier=ModelTier.FAST, llm=fast_llm,
        blackboard=bb, metrics=metrics, budget_monitor=budget,
    ))
    return agents, {"n_a": n_a, "n_b": n_b, "n_synth": n_synth, "n_critic": 1, "n_verifier": 1}


async def run_gaia_n(puzzle, idx, total, fast_llm, slow_llm, num_agents, log_dir):
    pid = puzzle["puzzle_id"]
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_gaia{num_agents}.jsonl")
    metrics = MetricsCollector()
    budget = BudgetMonitor(max_cost_per_problem=1.00, max_iterations=20, max_llm_calls=40)
    policy = Policy(max_iterations=20, stop_on_first_pass=True,
                    verification_strictness="all_tests_pass")

    agents, composition = build_gaia_agents(num_agents, bb, fast_llm, slow_llm, metrics, budget)
    loop = PuzzleEpisodeLoop(blackboard=bb, agents=agents,
                              metrics=metrics, policy=policy, budget_monitor=budget)

    t0 = time.time()
    result: PuzzleEpisodeResult = await loop.run_episode(puzzle)
    duration_s = time.time() - t0

    r = {
        "puzzle_id": pid,
        "condition": f"gaia_{num_agents}",
        "num_agents": num_agents,
        "condition_type": "gaia",
        "agent_composition": composition,
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

    status = f"{C.GREEN}PASS{C.END}" if result.passed else f"{C.FAIL}FAIL{C.END}"
    progress(idx, total, pid,
             f"{status}  n={num_agents}  cost=${result.cost_usd:.4f}")
    return r


# ===========================================================================
# Runner
# ===========================================================================

def load_puzzles(n=None):
    with open(DATA_PATH) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    if n:
        puzzles = puzzles[:n]
    return puzzles


async def run_condition_set(condition_type, num_agents, puzzles, fast_llm, slow_llm,
                              log_dir, checkpoint):
    results = []
    n = len(puzzles)
    completed_ids = set(checkpoint.get_completed_task_ids())

    for idx, puzzle in enumerate(puzzles, 1):
        pid = puzzle["puzzle_id"]
        ckpt_key = f"{condition_type}_{num_agents}/{pid}"
        if ckpt_key in completed_ids:
            progress(idx, n, pid, f"{C.YELLOW}(cached){C.END}")
            continue

        try:
            if condition_type == "homogeneous":
                r = await run_homogeneous(puzzle, idx, n, fast_llm, num_agents, log_dir)
            elif condition_type == "gaia":
                r = await run_gaia_n(puzzle, idx, n, fast_llm, slow_llm, num_agents, log_dir)
            else:
                raise ValueError(f"Unknown condition_type: {condition_type}")

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
                "puzzle_id": pid,
                "condition": f"{condition_type}_{num_agents}",
                "num_agents": num_agents,
                "condition_type": condition_type,
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
    parser = argparse.ArgumentParser(description="E8 Agent Scaling")
    parser.add_argument("--num-agents", type=str, default="all",
                        help="Agent count: 2, 4, 6, 8, or 'all'")
    parser.add_argument("--condition-type", choices=["homogeneous", "gaia", "all"],
                        default="all")
    parser.add_argument("--n_puzzles", type=int, default=None)
    return parser.parse_args()


async def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    puzzles = load_puzzles(args.n_puzzles)

    fast_llm = OpenAILLM(model=FAST_MODEL, tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model=SLOW_MODEL, tier=ModelTier.SLOW)

    agent_counts = [2, 4, 6, 8] if args.num_agents == "all" else [int(args.num_agents)]
    condition_types = (
        ["homogeneous", "gaia"] if args.condition_type == "all" else [args.condition_type]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for n_agents in agent_counts:
        for ctype in condition_types:
            hdr(f"E8 Agent Scaling — {ctype.upper()} n={n_agents}  ({len(puzzles)} puzzles)")

            log_dir = LOGS_DIR / f"{timestamp}_{ctype}_n{n_agents}"
            log_dir.mkdir(parents=True, exist_ok=True)

            ckpt_path = RESULTS_DIR / f"checkpoint_{ctype}_n{n_agents}_{timestamp}.json"
            checkpoint = CheckpointManager(ckpt_path)

            results = await run_condition_set(
                ctype, n_agents, puzzles, fast_llm, slow_llm, log_dir, checkpoint,
            )

            passed = sum(1 for r in results if r.get("passed"))
            total = len(results)
            cost = sum(r.get("cost_usd", 0) for r in results)
            acc = passed / total if total else 0
            print(f"\n{C.BOLD}── {ctype} n={n_agents} ──{C.END}  acc={acc:.1%}  cost=${cost:.4f}\n")

            key = f"{ctype}_n{n_agents}"
            all_results[key] = {
                "results": results,
                "summary": {
                    "condition_type": ctype,
                    "num_agents": n_agents,
                    "n_puzzles": total,
                    "n_passed": passed,
                    "accuracy": acc,
                    "total_cost_usd": cost,
                },
            }

    out_path = RESULTS_DIR / f"scaling_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{C.BOLD}Results saved to: {out_path}{C.END}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
