#!/usr/bin/env python3
"""Run the Fault Injection experiment (E9).

Claim: In real deployments, one agent's data source may be unreliable
(corrupted sensor, wrong document, hallucinating sub-agent). Can GAIA's
blackboard detect this and correct for it? Adding a DeductionAuditor +
TrustAwareSynthesizer enables fault-tolerant operation even when one expert
receives 30% corrupted clues.

Conditions:
  clean_gaia       — standard GAIA, all agents receive correct clues
  fault_standard   — standard GAIA, ONE expert receives 30% corrupted clues
  fault_gaia       — GAIA + Auditor + TrustAware Synth, ONE faulty expert

Folder structure:
  experiments/fault_injection/
    scripts/run_fault_injection.py  ← this file
    results/                         ← JSON + checkpoints
    logs/                            ← per-puzzle JSONL logs
"""

import argparse
import asyncio
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
from gaia.agents.puzzle.faulty_expert import FaultyExpertAgent
from gaia.agents.puzzle.deduction_auditor import DeductionAuditorAgent
from gaia.agents.puzzle.trust_synthesizer import TrustAwareSynthesizerAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.fault_injection_loop import (
    FaultInjectionEpisodeLoop, FaultInjectionEpisodeResult,
)
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "puzzle" / "puzzles.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results"
LOGS_DIR       = EXPERIMENT_DIR / "logs"

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


def hdr(text):
    print(f"\n{C.BOLD}{C.HEADER}{'='*80}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{text.center(80)}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'='*80}{C.END}\n")


def progress(i, n, pid, msg):
    print(f"{C.CYAN}[{i}/{n}]{C.END} {pid:25s} {msg}")


# ===========================================================================
# Noise clue generation
# ===========================================================================

def generate_noise_clues(puzzle: dict, n_noise: int, seed: int) -> List[str]:
    """Generate contradictory clues for a puzzle by perturbing the solution.

    Strategy: for each pair (person, attribute), if the correct value is X,
    write a noise clue stating the person has Y (some OTHER value from the
    attribute's possible value set). This guarantees logical contradiction
    with the true solution.

    Args:
        puzzle: puzzle dict with "solution" mapping person -> attr -> value.
        n_noise: number of noise clues to generate.
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    solution = puzzle["solution"]
    people = list(solution.keys())
    attributes_in_use = {}
    for person, attrs in solution.items():
        for attr, val in attrs.items():
            attributes_in_use.setdefault(attr, set()).add(val)
    # Convert to lists for sampling
    attributes_pool = {attr: sorted(vs) for attr, vs in attributes_in_use.items()}

    noise_clues = []
    attempts = 0
    while len(noise_clues) < n_noise and attempts < n_noise * 10:
        attempts += 1
        person = rng.choice(people)
        attr = rng.choice(list(attributes_pool.keys()))
        correct_val = solution[person][attr]
        wrong_choices = [v for v in attributes_pool[attr] if v != correct_val]
        if not wrong_choices:
            continue
        wrong_val = rng.choice(wrong_choices)

        # Format as a natural-language clue (matches puzzles.json style)
        if attr == "job":
            clue = f"{person} is the {wrong_val}."
        elif attr == "pet":
            clue = f"{person} keeps a {wrong_val}."
        elif attr == "drink":
            clue = f"{person} drinks {wrong_val}."
        else:
            clue = f"{person}'s {attr} is {wrong_val}."

        if clue not in noise_clues:
            noise_clues.append(clue)

    return noise_clues


# ===========================================================================
# Conditions
# ===========================================================================

async def run_clean_gaia(puzzle, idx, total, fast_llm, slow_llm, budget, log_dir):
    """Standard GAIA, all experts get correct clues."""
    pid = puzzle["puzzle_id"]
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_clean_gaia.jsonl")
    metrics = MetricsCollector()
    policy = Policy(max_iterations=20, stop_on_first_pass=True)

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

    loop = FaultInjectionEpisodeLoop(blackboard=bb, agents=agents,
                                       metrics=metrics, policy=policy, budget_monitor=budget)
    result = await loop.run_episode(puzzle)
    return _to_dict(result, "clean_gaia")


async def run_fault_standard(puzzle, idx, total, fast_llm, slow_llm, budget, log_dir, seed):
    """One expert is faulty (30% corruption), NO auditor."""
    pid = puzzle["puzzle_id"]
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_fault_standard.jsonl")
    metrics = MetricsCollector()
    policy = Policy(max_iterations=20, stop_on_first_pass=True)

    noise_clues = generate_noise_clues(puzzle, n_noise=8, seed=seed)

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
        # Faulty expert in partition B
        FaultyExpertAgent(
            name="FaultyExpert-B", partition="B",
            noise_clues=noise_clues, corruption_rate=0.3, seed=seed,
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
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

    loop = FaultInjectionEpisodeLoop(blackboard=bb, agents=agents,
                                       metrics=metrics, policy=policy, budget_monitor=budget)
    result = await loop.run_episode(puzzle)
    return _to_dict(result, "fault_standard", noise_clues=noise_clues)


async def run_fault_gaia(puzzle, idx, total, fast_llm, slow_llm, budget, log_dir, seed,
                          partial_trust=False, condition_label="fault_gaia"):
    """One expert is faulty (30% corruption), Auditor + TrustAware Synth ENABLED.

    partial_trust=False : agent-level down-weighting (naive defense).
    partial_trust=True  : clue-level skepticism (principled fix) — keeps a
        flagged expert's uncontradicted necessary clues.
    """
    pid = puzzle["puzzle_id"]
    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_{condition_label}.jsonl")
    metrics = MetricsCollector()
    policy = Policy(max_iterations=20, stop_on_first_pass=True)

    noise_clues = generate_noise_clues(puzzle, n_noise=8, seed=seed)

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
        FaultyExpertAgent(
            name="FaultyExpert-B", partition="B",
            noise_clues=noise_clues, corruption_rate=0.3, seed=seed,
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # NEW: Auditor agent
        DeductionAuditorAgent(
            name="DeductionAuditor",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # NEW: Trust-aware synthesizers replace standard ones
        TrustAwareSynthesizerAgent(
            name="TrustAwareSynth-1",
            people=DEFAULT_PEOPLE, attributes=DEFAULT_ATTRIBUTES,
            partial_trust=partial_trust,
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        TrustAwareSynthesizerAgent(
            name="TrustAwareSynth-2",
            people=DEFAULT_PEOPLE, attributes=DEFAULT_ATTRIBUTES,
            partial_trust=partial_trust,
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        PuzzleCriticAgent(name="Critic",
                          tier=ModelTier.FAST, llm=fast_llm,
                          blackboard=bb, metrics=metrics, budget_monitor=budget),
        PuzzleVerifierAgent(name="Verifier",
                            tier=ModelTier.FAST, llm=fast_llm,
                            blackboard=bb, metrics=metrics, budget_monitor=budget),
    ]

    loop = FaultInjectionEpisodeLoop(blackboard=bb, agents=agents,
                                       metrics=metrics, policy=policy, budget_monitor=budget)
    result = await loop.run_episode(puzzle)
    return _to_dict(result, condition_label, noise_clues=noise_clues)


def _to_dict(result: FaultInjectionEpisodeResult, condition: str, noise_clues=None) -> dict:
    return {
        "puzzle_id": result.puzzle_id,
        "condition": condition,
        "passed": result.passed,
        "proposed_solution": result.proposed_solution,
        "fault_injected": result.fault_injected,
        "auditor_used": result.auditor_used,
        "auditor_flagged_faulty_agent": result.auditor_flagged_faulty_agent,
        "auditor_suspect_id": result.auditor_suspect_id,
        "real_faulty_agent_id": result.real_faulty_agent_id,
        "trust_scores": result.trust_scores,
        "n_contradictions_found": result.n_contradictions_found,
        "num_expert_deductions": result.num_expert_deductions,
        "conflict_detected": result.conflict_detected,
        "conflict_resolved": result.conflict_resolved,
        "cost_usd": result.cost_usd,
        "duration_s": result.duration_s,
        "phase_timings": result.phase_timings,
        "noise_clues": noise_clues,
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


async def run_condition(condition, puzzles, fast_llm, slow_llm, log_dir, checkpoint, seed):
    results = []
    budget = BudgetMonitor(max_cost_per_problem=1.00, max_iterations=20, max_llm_calls=40)
    n = len(puzzles)
    completed_ids = set(checkpoint.get_completed_task_ids())

    for idx, puzzle in enumerate(puzzles, 1):
        pid = puzzle["puzzle_id"]
        ckpt_key = f"{condition}/{pid}"
        if ckpt_key in completed_ids:
            progress(idx, n, pid, f"{C.YELLOW}(cached){C.END}")
            continue

        try:
            budget.reset()
            if condition == "clean_gaia":
                r = await run_clean_gaia(puzzle, idx, n, fast_llm, slow_llm, budget, log_dir)
            elif condition == "fault_standard":
                r = await run_fault_standard(puzzle, idx, n, fast_llm, slow_llm, budget, log_dir, seed)
            elif condition == "fault_gaia":
                r = await run_fault_gaia(puzzle, idx, n, fast_llm, slow_llm, budget, log_dir, seed,
                                         partial_trust=False, condition_label="fault_gaia")
            elif condition == "fault_gaia_partial":
                r = await run_fault_gaia(puzzle, idx, n, fast_llm, slow_llm, budget, log_dir, seed,
                                         partial_trust=True, condition_label="fault_gaia_partial")
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
            extra = ""
            if r.get("auditor_used"):
                ok_flag = "✓" if r.get("auditor_flagged_faulty_agent") else "✗"
                extra = f" auditor[{ok_flag}]"
            progress(idx, n, pid,
                     f"{status}{extra}  cost=${r.get('cost_usd', 0):.4f}")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = {"puzzle_id": pid, "condition": condition,
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
    parser = argparse.ArgumentParser(description="E9 Fault Injection")
    parser.add_argument(
        "--condition",
        choices=["clean_gaia", "fault_standard", "fault_gaia",
                 "fault_gaia_partial", "all"],
        default="all",
    )
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

    conditions = (
        ["clean_gaia", "fault_standard", "fault_gaia", "fault_gaia_partial"]
        if args.condition == "all" else [args.condition]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for condition in conditions:
        hdr(f"E9 Fault Injection — {condition.upper()}  ({len(puzzles)} puzzles)")

        log_dir = LOGS_DIR / f"{timestamp}_{condition}"
        log_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = RESULTS_DIR / f"checkpoint_{condition}_{timestamp}.json"
        checkpoint = CheckpointManager(ckpt_path)

        results = await run_condition(
            condition, puzzles, fast_llm, slow_llm, log_dir, checkpoint, args.seed,
        )

        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        cost = sum(r.get("cost_usd", 0) for r in results)
        acc = passed / total if total else 0

        # Stats specific to E9: how often did the auditor correctly identify the fault?
        flagged_correctly = 0
        if condition in ("fault_gaia", "fault_gaia_partial"):
            flagged_correctly = sum(1 for r in results if r.get("auditor_flagged_faulty_agent"))

        all_results[condition] = {
            "results": results,
            "summary": {
                "condition": condition,
                "n_puzzles": total,
                "n_passed": passed,
                "accuracy": acc,
                "total_cost_usd": cost,
                "auditor_correctly_flagged_faulty": flagged_correctly if condition in ("fault_gaia", "fault_gaia_partial") else None,
            },
        }
        print(f"\n{C.BOLD}── {condition} ──{C.END}  acc={acc:.1%}  cost=${cost:.4f}")
        if condition in ("fault_gaia", "fault_gaia_partial"):
            print(f"  Auditor correctly flagged faulty agent in {flagged_correctly}/{total} runs")

    out_path = RESULTS_DIR / f"fault_injection_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{C.BOLD}Results saved to: {out_path}{C.END}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
