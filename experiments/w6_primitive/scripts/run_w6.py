#!/usr/bin/env python3
"""W6 — Blackboard-primitive ablation (information content of the signal).

NX1/C2-2 already isolated *whether* conflict-as-task helps (it does, +15.4pp,
dissenter-gated). W6 goes deeper: is the active ingredient the *typed,
content-rich* CONFLICT signal, or merely "knowing a disagreement exists"?
On the 13 E3 traps (2 misled + 1 clean), GAIA pool, we ablate ONLY what the
reconciler receives:

  full     — typed CONFLICT carries the conflict_summary (who-said-what) +
             all solver chains (the real GAIA).
  untyped  — reconciler told only "a disagreement exists" (no summary, no
             attribution) + chains. Tests value of the TYPED content.
  blind    — reconciler gets the chains but NO signal/summary at all and is
             just asked to "pick the best" (≈ plain re-read).
  none     — no reconciler; majority of the 3 solver answers (lower bound).

If accuracy drops sharply full→untyped→blind→none, the *typed signal content*
(not just redundancy) is the load-bearing primitive.
"""
import asyncio, json, os, sys, logging
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
_env = ROOT/".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

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
from gaia.prompts.math.reconciler import MathReconcilerPrompts

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"figures"  # reuse dir; logs unused
CSYS = MathSolverPrompts.SYSTEM
def cusr(q): return MathSolverPrompts.format_user(q)
MSYS = MisledSolverPrompts.SYSTEM
def musr(q, h): return MisledSolverPrompts.format_user(q, h)
FAST, SLOW = "gpt-4.1-nano", "gpt-4.1"


async def call(a, sysm, um, t=0.0):
    r = await a.call_llm([{"role": "system", "content": sysm},
                          {"role": "user", "content": um}], temperature=t)
    return r, extract_final_answer(r)


async def episode(p, ablation, fast, slow, bud):
    q, truth, hint = p["question"], p["answer"], p["misleading_hint"]
    bb = Blackboard(log_file=Path("/tmp/_w6.jsonl"))
    m = MetricsCollector()
    A = [MisledSolverAgent(misled_index=0, name="M0", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
         MisledSolverAgent(misled_index=1, name="M1", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
         MathSolverAgent(solver_index=0, name="C0", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud)]
    r0 = await asyncio.gather(
        call(A[0], MSYS, musr(q, hint)), call(A[1], MSYS, musr(q, hint)),
        call(A[2], CSYS, cusr(q)))
    chains = [x[0] for x in r0]; ans = [x[1] for x in r0]
    valid = [a for a in ans if a is not None]
    if len(set(valid)) <= 1:                       # no conflict
        fin = Counter(valid).most_common(1)[0][0] if valid else None
        return fin == truth, bud.current_cost
    if ablation == "none":
        fin = Counter(valid).most_common(1)[0][0]  # majority (lower bound)
        return fin == truth, bud.current_cost
    rec = MathSolverAgent(solver_index=8, name="Rec", tier=ModelTier.SLOW,
        llm=slow, blackboard=bb, metrics=m, budget_monitor=bud)
    rp = MathReconcilerPrompts()
    names = ["Solver-1", "Solver-2", "Solver-3"]
    if ablation == "full":
        summary = ", ".join(f"{names[i]}={ans[i]}" for i in range(3))
        um = rp.format_user(q, summary, [(names[i], chains[i]) for i in range(3)])
    elif ablation == "untyped":
        um = rp.format_user(q, "A disagreement exists (no attribution).",
                            [(names[i], chains[i]) for i in range(3)])
    else:  # blind
        um = rp.format_user(q, "",
                            [("Write-up", chains[i]) for i in range(3)])
    _, fin = await call(rec, rp.SYSTEM, um)
    return fin == truth, bud.current_cost


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for ab in ("full", "untyped", "blind", "none"):
        npass = 0; cost = 0.0; recs = []
        for p in probs:
            bud = BudgetMonitor(max_cost_per_problem=0.6, max_iterations=20,
                                max_llm_calls=40)
            try:
                ok, c = await asyncio.wait_for(
                    episode(p, ab, fast, slow, bud), timeout=150)
            except Exception:
                ok, c = False, bud.current_cost
            npass += int(ok); cost += c
            recs.append({"problem_id": p["problem_id"], "passed": ok})
            print(f"[{ab:8s}] {p['problem_id']:22s} {'PASS' if ok else 'FAIL'}")
        out[ab] = {"summary": {"ablation": ab, "n": len(probs),
            "accuracy": npass/len(probs), "total_cost_usd": cost}}
        print(f"== {ab}: acc={out[ab]['summary']['accuracy']:.0%} "
              f"cost=${cost:.3f} ==")
    json.dump(out, open(RES/f"w6_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved w6_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
