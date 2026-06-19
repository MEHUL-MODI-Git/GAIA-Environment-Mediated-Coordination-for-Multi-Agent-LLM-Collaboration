#!/usr/bin/env python3
"""Run the Asymmetric Information Puzzle experiment.

Three experimental conditions (for paper ablation):
  --condition single    : 1 agent sees ALL 12 clues (baseline)
  --condition isolated  : 4 experts with split clues, NO blackboard sharing (ablation)
  --condition gaia      : full 8-agent GAIA system with blackboard (main result)
  --condition all       : run all three in sequence (default)

Models:
  Fast tier (Experts, Critic, Verifier): gpt-4.1-mini
  Slow tier (Synthesizers): gpt-4.1

Folder structure (self-contained):
  experiments/puzzle/
    scripts/run_puzzle.py       ← this file
    results/                    ← JSON results + checkpoints
    logs/                       ← per-puzzle JSONL logs

Usage:
  python experiments/puzzle/scripts/run_puzzle.py
  python experiments/puzzle/scripts/run_puzzle.py --condition gaia --puzzles 5
  python experiments/puzzle/scripts/run_puzzle.py --condition single
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Project root (scripts/ → puzzle/ → experiments/ → GAIA root) ──────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Auto-load .env ─────────────────────────────────────────────────────────
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
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.puzzle_loop import PuzzleEpisodeLoop, PuzzleEpisodeResult
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


# ── Paths ──────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_PATH      = PROJECT_ROOT / "data" / "puzzle" / "puzzles.json"
RESULTS_DIR    = EXPERIMENT_DIR / "results"
LOGS_DIR       = EXPERIMENT_DIR / "logs"

# ── Models ─────────────────────────────────────────────────────────────────
FAST_MODEL = "gpt-4.1-mini"   # Experts, Critic, Verifier
SLOW_MODEL = "gpt-4.1"        # Synthesizers


# ── Terminal colours ────────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    FAIL    = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"


def hdr(text: str):
    print(f"\n{C.BOLD}{C.HEADER}{'='*80}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{text.center(80)}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'='*80}{C.END}\n")


def progress(i: int, n: int, pid: str, msg: str):
    print(f"{C.CYAN}[{i}/{n}]{C.END} {pid:25s} {msg}")


def stats(results: List[dict]):
    passed = sum(1 for r in results if r.get("passed"))
    total  = len(results)
    cost   = sum(r.get("cost_usd", 0) for r in results)
    conflicts = sum(1 for r in results if r.get("conflict_detected"))
    rate   = passed / total if total else 0
    print(f"\n{C.BOLD}── Statistics ──{C.END}")
    print(f"  Completed : {total}")
    print(f"  Passed    : {C.GREEN}{passed}{C.END}")
    print(f"  Failed    : {C.FAIL}{total - passed}{C.END}")
    print(f"  Accuracy  : {C.BOLD}{rate:.1%}{C.END}")
    print(f"  Conflicts : {conflicts} ({conflicts/total:.1%} of puzzles)")
    print(f"  Cost      : ${cost:.4f}\n")


# ===========================================================================
# Single-agent baseline (one agent sees ALL clues)
# ===========================================================================

async def run_single_agent(
    puzzle: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
) -> dict:
    """One agent sees ALL 12 clues and directly synthesizes a solution.

    This is the oracle baseline: tests whether a capable LLM can solve a
    4×3 logic-grid puzzle when given complete information in a single pass.
    Uses the Synthesizer prompt (which outputs Alice: job=X, pet=Y, drink=Z)
    so the solution is parseable.
    """
    from gaia.agents.puzzle.synthesizer import parse_solution_from_text
    from gaia.agents.puzzle.puzzle_verifier import proposed_matches_ground_truth
    from gaia.prompts.puzzle.synthesizer import SynthesizerPrompts
    from gaia.utils.metrics import MetricsCollector

    pid = puzzle["puzzle_id"]
    budget.reset()

    ground_truth   = puzzle["solution"]
    all_clues_texts = [c["text"] for c in puzzle["all_clues"]]

    # Format all 12 clues as a single "expert" input so the synthesizer sees
    # the full picture — no information asymmetry.
    prompts = SynthesizerPrompts()
    all_deductions = [
        (
            "SingleExpert",
            "ALL",
            "All clues (no partition split):\n" + "\n".join(f"  {c}" for c in all_clues_texts),
        )
    ]

    messages = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "user", "content": prompts.format_user(all_deductions)},
    ]

    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_single.jsonl")
    metrics = MetricsCollector()
    agent = SynthesizerAgent(
        name="SingleSynthesizer",
        tier=ModelTier.SLOW,
        llm=fast_llm,  # Use fast model to keep cost low
        blackboard=bb,
        metrics=metrics,
        budget_monitor=budget,
    )

    response = await agent.call_llm(messages, temperature=0.1)
    proposed = parse_solution_from_text(response)

    passed = False
    if proposed:
        passed, _ = proposed_matches_ground_truth(proposed, ground_truth)

    return {
        "puzzle_id": pid,
        "condition": "single",
        "passed": passed,
        "proposed_solution": proposed,
        "cost_usd": budget.current_cost,
        "duration_s": 0.0,
        "conflict_detected": False,
        "error": None,
    }


# ===========================================================================
# Isolated condition (experts cannot share; synthesizer sees raw clues only)
# ===========================================================================

async def run_isolated(
    puzzle: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
) -> dict:
    """No-sharing ablation: each synthesizer sees ONLY its own partition's clues.

    Design:
    - Synthesizer-A receives only Partition A clues (6 clues)
    - Synthesizer-B receives only Partition B clues (6 clues)
    - They run in parallel but CANNOT see each other's information
    - Final answer = majority vote (both should agree if both correct;
      if they disagree, pick the more "complete" solution)

    This simulates a world with NO shared blackboard.  Since each partition
    alone is mathematically ambiguous (2 consistent solutions), a lone
    synthesizer cannot determine the unique answer → expected accuracy ≈ 0%.
    """
    from gaia.agents.puzzle.synthesizer import parse_solution_from_text
    from gaia.agents.puzzle.puzzle_verifier import proposed_matches_ground_truth
    from gaia.prompts.puzzle.synthesizer import SynthesizerPrompts
    from gaia.utils.metrics import MetricsCollector

    pid = puzzle["puzzle_id"]
    budget.reset()

    ground_truth  = puzzle["solution"]
    clues_a_texts = [c["text"] for c in puzzle["clues_a"]]
    clues_b_texts = [c["text"] for c in puzzle["clues_b"]]

    prompts = SynthesizerPrompts()
    safe_id = pid.replace("/", "_")

    # Synthesizer-A: only sees Partition A
    deductions_a = [
        ("ExpertA", "A", "Clues from Partition A only:\n" + "\n".join(f"  {c}" for c in clues_a_texts)),
    ]
    # Synthesizer-B: only sees Partition B
    deductions_b = [
        ("ExpertB", "B", "Clues from Partition B only:\n" + "\n".join(f"  {c}" for c in clues_b_texts)),
    ]

    bb_a = Blackboard(log_file=log_dir / f"{safe_id}_isolated_a.jsonl")
    bb_b = Blackboard(log_file=log_dir / f"{safe_id}_isolated_b.jsonl")
    metrics_a, metrics_b = MetricsCollector(), MetricsCollector()

    synth_a = SynthesizerAgent(
        name="IsolatedSynth-A",
        tier=ModelTier.SLOW,
        llm=slow_llm,
        blackboard=bb_a,
        metrics=metrics_a,
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )
    synth_b = SynthesizerAgent(
        name="IsolatedSynth-B",
        tier=ModelTier.SLOW,
        llm=slow_llm,
        blackboard=bb_b,
        metrics=metrics_b,
        budget_monitor=BudgetMonitor(max_cost_per_problem=0.50, max_iterations=5, max_llm_calls=5),
    )

    async def call_synth(agent, deductions):
        msgs = [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.format_user(deductions)},
        ]
        resp = await agent.call_llm(msgs, temperature=0.1)
        return parse_solution_from_text(resp)

    # Run in parallel — no shared state
    proposed_a, proposed_b = await asyncio.gather(
        call_synth(synth_a, deductions_a),
        call_synth(synth_b, deductions_b),
    )

    # Score both; use Synth-A's answer as primary (ties broken by A)
    # In practice both should fail since each partition is ambiguous
    passed_a = bool(proposed_a and proposed_matches_ground_truth(proposed_a, ground_truth)[0])
    passed_b = bool(proposed_b and proposed_matches_ground_truth(proposed_b, ground_truth)[0])
    proposed = proposed_a or proposed_b
    passed   = passed_a or passed_b  # True only if either independently solved it

    return {
        "puzzle_id": pid,
        "condition": "isolated",
        "passed": passed,
        "proposed_solution": proposed,
        "cost_usd": budget.current_cost,
        "duration_s": 0.0,
        "conflict_detected": False,
        "isolated_a_passed": passed_a,
        "isolated_b_passed": passed_b,
        "error": None,
    }


# ===========================================================================
# GAIA condition (full 8-agent system)
# ===========================================================================

async def run_gaia(
    puzzle: dict,
    idx: int,
    total: int,
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    budget: BudgetMonitor,
    log_dir: Path,
    max_cost: float = 0.50,
) -> dict:
    """Full GAIA system: 2 Expert-A + 2 Expert-B + 2 Synthesizer + 1 Critic + 1 Verifier."""
    pid = puzzle["puzzle_id"]
    budget.reset()
    budget.max_cost_per_problem = max_cost

    safe_id = pid.replace("/", "_")
    bb = Blackboard(log_file=log_dir / f"{safe_id}_gaia.jsonl")
    metrics = MetricsCollector()

    policy = Policy(
        max_iterations=20,
        stop_on_first_pass=True,
        verification_strictness="all_tests_pass",
    )

    agents = [
        # Partition A experts (2 agents, different temperatures)
        ExpertAgent(
            name="Expert-A-1", partition="A",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        ExpertAgent(
            name="Expert-A-2", partition="A",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Partition B experts (2 agents, different temperatures)
        ExpertAgent(
            name="Expert-B-1", partition="B",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        ExpertAgent(
            name="Expert-B-2", partition="B",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Synthesizers (2 agents, use the powerful model)
        SynthesizerAgent(
            name="Synthesizer-1",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        SynthesizerAgent(
            name="Synthesizer-2",
            tier=ModelTier.SLOW, llm=slow_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Critic (detects disagreements between synthesizers)
        PuzzleCriticAgent(
            name="Critic",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
        # Verifier (Python solver — ground truth)
        PuzzleVerifierAgent(
            name="Verifier",
            tier=ModelTier.FAST, llm=fast_llm,
            blackboard=bb, metrics=metrics, budget_monitor=budget,
        ),
    ]

    loop = PuzzleEpisodeLoop(
        blackboard=bb,
        agents=agents,
        metrics=metrics,
        policy=policy,
        budget_monitor=budget,
    )

    import time
    t0 = time.time()
    result: PuzzleEpisodeResult = await loop.run_episode(puzzle)
    duration_s = time.time() - t0

    return {
        "puzzle_id": pid,
        "condition": "gaia",
        "passed": result.passed,
        "proposed_solution": result.proposed_solution,
        "cost_usd": result.cost_usd,
        "duration_s": duration_s,
        "conflict_detected": result.conflict_detected,
        "conflict_resolved": result.conflict_resolved,
        "num_expert_deductions": result.num_expert_deductions,
        "num_synthesis_artifacts": result.num_synthesis_artifacts,
        "phase_timings": result.phase_timings,
        "error": result.error,
    }


# ===========================================================================
# Main
# ===========================================================================

async def run_condition(
    condition: str,
    puzzles: List[dict],
    fast_llm: OpenAILLM,
    slow_llm: OpenAILLM,
    budget: BudgetMonitor,
    checkpoint: CheckpointManager,
    log_dir: Path,
    max_puzzles: Optional[int] = None,
) -> List[dict]:
    """Run one experimental condition over the full puzzle set."""
    hdr(f"Condition: {condition.upper()}")
    completed_ids = set(checkpoint.get_completed_task_ids())
    results = []
    run_puzzles = puzzles[:max_puzzles] if max_puzzles else puzzles

    for i, puzzle in enumerate(run_puzzles):
        pid = puzzle["puzzle_id"]
        ckpt_key = f"{condition}/{pid}"

        if ckpt_key in completed_ids:
            progress(i + 1, len(run_puzzles), pid, f"{C.YELLOW}(skipped — already done){C.END}")
            continue

        print(f"\n{C.BOLD}── [{i+1}/{len(run_puzzles)}] {pid} ──{C.END}")

        try:
            if condition == "single":
                result = await run_single_agent(puzzle, i, len(run_puzzles), fast_llm, budget, log_dir)
            elif condition == "isolated":
                result = await run_isolated(puzzle, i, len(run_puzzles), fast_llm, slow_llm, budget, log_dir)
            else:  # gaia
                result = await run_gaia(puzzle, i, len(run_puzzles), fast_llm, slow_llm, budget, log_dir)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  {C.FAIL}ERROR: {error_msg}{C.END}")
            traceback.print_exc()
            result = {
                "puzzle_id": pid,
                "condition": condition,
                "passed": False,
                "cost_usd": budget.current_cost,
                "error": error_msg,
            }

        results.append(result)
        status = f"{C.GREEN}✓ PASS{C.END}" if result["passed"] else f"{C.FAIL}✗ FAIL{C.END}"
        cost = result.get("cost_usd", 0)
        print(f"  {status}  cost=${cost:.4f}")

        # Checkpoint
        checkpoint.add_result(
            task_id=ckpt_key,
            passed=result["passed"],
            iterations=result.get("iterations", 1),
            cost_usd=result.get("cost_usd", 0),
            duration_s=result.get("duration_s", 0),
            stop_reason="passed" if result["passed"] else "failed",
            num_conflicts=1 if result.get("conflict_detected") else 0,
            error=result.get("error"),
        )

        if (i + 1) % 5 == 0:
            stats(results)

    return results


async def main():
    parser = argparse.ArgumentParser(description="GAIA Asymmetric Puzzle Experiment")
    parser.add_argument("--condition", choices=["single", "isolated", "gaia", "all"],
                        default="all", help="Experimental condition to run")
    parser.add_argument("--puzzles", type=int, default=None,
                        help="Limit to first N puzzles (default: all 20)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpointing (re-run everything)")
    args = parser.parse_args()

    hdr("GAIA Asymmetric Information Puzzle Experiment")

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # API key check
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{C.FAIL}❌ OPENAI_API_KEY not set{C.END}")
        return 1

    # Load puzzles
    print(f"Loading puzzles from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    puzzles = data["puzzles"]
    print(f"✓ Loaded {len(puzzles)} puzzles\n")

    if args.puzzles:
        puzzles = puzzles[:args.puzzles]
        print(f"Limiting to {len(puzzles)} puzzles (--puzzles {args.puzzles})\n")

    # Setup dirs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_DIR / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = RESULTS_DIR / f"puzzle_{timestamp}.checkpoint.json"
    output_path     = RESULTS_DIR / f"puzzle_{timestamp}.results.json"

    checkpoint = CheckpointManager(checkpoint_path)
    checkpoint.start_run(total_problems=len(puzzles) * (3 if args.condition == "all" else 1))

    # Initialize LLMs
    print(f"Models: fast={FAST_MODEL}  slow={SLOW_MODEL}")
    fast_llm = OpenAILLM(model=FAST_MODEL, tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model=SLOW_MODEL, tier=ModelTier.SLOW)
    print("✓ LLMs initialized\n")

    budget = BudgetMonitor(
        max_cost_per_problem=0.50,
        max_iterations=25,
        max_llm_calls=40,
    )

    conditions = (
        ["single", "isolated", "gaia"]
        if args.condition == "all"
        else [args.condition]
    )

    all_results: Dict[str, List[dict]] = {}

    try:
        for condition in conditions:
            cond_results = await run_condition(
                condition=condition,
                puzzles=puzzles,
                fast_llm=fast_llm,
                slow_llm=slow_llm,
                budget=budget,
                checkpoint=checkpoint,
                log_dir=log_dir,
                max_puzzles=args.puzzles,
            )
            all_results[condition] = cond_results

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}⚠ Interrupted — partial results saved{C.END}\n")

    # Final summary
    hdr("FINAL RESULTS")
    for condition, results in all_results.items():
        if results:
            print(f"\n  {C.BOLD}{condition.upper()}{C.END}:")
            stats(results)

    # Save
    output = {
        "run_timestamp": timestamp,
        "models": {"fast": FAST_MODEL, "slow": SLOW_MODEL},
        "total_puzzles": len(puzzles),
        "conditions_run": conditions,
        "results_by_condition": all_results,
        "summary": {
            condition: {
                "accuracy": (
                    sum(1 for r in res if r.get("passed")) / len(res) if res else 0
                ),
                "total_cost": sum(r.get("cost_usd", 0) for r in res),
                "conflict_rate": (
                    sum(1 for r in res if r.get("conflict_detected")) / len(res) if res else 0
                ),
            }
            for condition, res in all_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{C.GREEN}✓ Results saved: {output_path}{C.END}")
    print(f"{C.GREEN}✓ Episode logs:  {log_dir}/{C.END}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted{C.END}")
        sys.exit(130)
