#!/usr/bin/env python3
"""Validate the misled-solver design for E3.

For E3's mechanism to be testable we need, per problem:
  - 2 misled solvers (same hint) BOTH produce the SAME wrong answer
    → a deterministic correlated failure (wrong 2/3 majority)
  - 1 clean solver produces the CORRECT answer (the dissenter)

This script measures exactly that and writes a report so we can drop/replace
any problem whose hint does not reliably mislead BEFORE the full E3 run.

Usage:
  python experiments/correlated_failure/scripts/validate_traps.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

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
from gaia.agents.math import MathSolverAgent
from gaia.agents.math.math_solver import extract_final_answer
from gaia.agents.math.misled_solver import MisledSolverAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts

DATA_PATH = PROJECT_ROOT / "data" / "gsm8k" / "correlated_failure_problems.json"
OUT_PATH = Path(__file__).parent.parent / "results" / "trap_validation.json"
FAST_MODEL = "gpt-4.1-nano"


async def main():
    with open(DATA_PATH) as f:
        problems = json.load(f)

    llm = OpenAILLM(model=FAST_MODEL, tier=ModelTier.FAST)
    clean_prompts = MathSolverPrompts()
    misled_prompts = MisledSolverPrompts()

    bb = Blackboard(log_file=Path("/tmp/trap_validation.jsonl"))
    metrics = MetricsCollector()
    budget = BudgetMonitor(max_cost_per_problem=1.0, max_iterations=80, max_llm_calls=200)

    clean = MathSolverAgent(solver_index=0, name="Clean", tier=ModelTier.FAST,
                            llm=llm, blackboard=bb, metrics=metrics, budget_monitor=budget)
    misled = MisledSolverAgent(misled_index=0, name="Misled", tier=ModelTier.FAST,
                               llm=llm, blackboard=bb, metrics=metrics, budget_monitor=budget)

    async def solve_clean(q):
        msgs = [{"role": "system", "content": clean_prompts.SYSTEM},
                {"role": "user", "content": clean_prompts.format_user(q)}]
        return extract_final_answer(await clean.call_llm(msgs, temperature=0.0))

    async def solve_misled(q, hint):
        msgs = [{"role": "system", "content": misled_prompts.SYSTEM},
                {"role": "user", "content": misled_prompts.format_user(q, hint)}]
        return extract_final_answer(await misled.call_llm(msgs, temperature=0.0))

    report = []
    n_good = 0  # 2 misled agree+wrong AND clean right
    print(f"Validating misled design on {FAST_MODEL} ({len(problems)} problems)...\n")
    for i, p in enumerate(problems, 1):
        q, truth, hint = p["question"], p["answer"], p["misleading_hint"]
        # Run misled twice (two independent solvers, same hint) + clean once
        m0, m1, c = await asyncio.gather(
            solve_misled(q, hint), solve_misled(q, hint), solve_clean(q)
        )
        misled_agree_wrong = (m0 == m1 and m0 is not None and m0 != truth)
        clean_right = (c == truth)
        good = misled_agree_wrong and clean_right
        if good:
            n_good += 1

        flag = ("✓ GOOD" if good else
                ("~ clean also wrong" if not clean_right else
                 "✗ misled not correlated/wrong"))
        print(f"[{i:2d}/{len(problems)}] {p['problem_id']:22s} "
              f"truth={truth:>5} misled=({m0},{m1}) clean={c}  | {flag}")

        report.append({
            "problem_id": p["problem_id"], "category": p["category"],
            "truth": truth, "expected_wrong": p["common_wrong_answer"],
            "misled_0": m0, "misled_1": m1, "clean": c,
            "misled_agree_wrong": misled_agree_wrong,
            "clean_correct": clean_right,
            "good": good,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "model": FAST_MODEL, "design": "misled",
            "n_problems": len(problems), "n_good": n_good,
            "total_cost_usd": budget.current_cost, "report": report,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"GOOD (2 misled agree+wrong, clean right): {n_good}/{len(problems)}")
    print(f"Cost: ${budget.current_cost:.4f}  Report: {OUT_PATH}")
    print(f"{'='*70}")
    if n_good >= 10:
        print(f"\n✓ {n_good}/15 — strong, clean E3 result expected.")
    elif n_good >= 6:
        print(f"\n~ {n_good}/15 — usable; filter to good problems for the headline figure.")
    else:
        print(f"\n⚠️  Only {n_good}/15 — hints need strengthening.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
