#!/usr/bin/env python3
"""W9 — MAST failure-injection matrix (interventional / causal NX2).

NX2 was OBSERVATIONAL (classify failures that happened). W9 is
INTERVENTIONAL: deliberately inject a specific MAST failure mode into GAIA's
E3 pipeline and measure how much accuracy it costs — i.e. how well GAIA's
structure CONTAINS each injected mode. Same 13 traps (2 misled + 1 clean).

Injected modes (chosen because they are cleanly, faithfully simulable):
  none         — baseline GAIA (control).
  FM-2.5       — Ignored peer input: the reconciler is DENIED the clean
                 solver's chain (only the 2 misled chains reach it).
  FM-3.2       — No/incomplete verification: skip the reconciler entirely,
                 emit the aggregator's majority (the misled wrong answer).
  FM-1.3       — Step repetition: feed the reconciler the SAME misled chain
                 three times (duplicate, no clean signal).
  FM-2.6       — Reasoning-action mismatch: reconciler reasons normally but
                 its final line is overwritten with the misled answer.

Containment = how close to baseline accuracy GAIA stays under each injection.
Modes GAIA structurally resists → small drop; modes it cannot → large drop.
This is the causal complement to NX2's mode map.
"""
import asyncio, json, os, re, sys, logging
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
CSYS = MathSolverPrompts.SYSTEM
def cusr(q): return MathSolverPrompts.format_user(q)
MSYS = MisledSolverPrompts.SYSTEM
def musr(q, h): return MisledSolverPrompts.format_user(q, h)
FAST, SLOW = "gpt-4.1-nano", "gpt-4.1"


async def call(a, s, u, t=0.0):
    r = await a.call_llm([{"role": "system", "content": s},
                          {"role": "user", "content": u}], temperature=t)
    return r, extract_final_answer(r)


async def episode(p, mode, fast, slow, bud):
    q, truth, hint, tgt = (p["question"], p["answer"],
                           p["misleading_hint"], p["common_wrong_answer"])
    bb = Blackboard(log_file=Path("/tmp/_w9.jsonl")); m = MetricsCollector()
    A = [MisledSolverAgent(misled_index=0, name="M0", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
         MisledSolverAgent(misled_index=1, name="M1", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
         MathSolverAgent(solver_index=0, name="C0", tier=ModelTier.FAST,
            llm=fast, blackboard=bb, metrics=m, budget_monitor=bud)]
    r0 = await asyncio.gather(
        call(A[0], MSYS, musr(q, hint)), call(A[1], MSYS, musr(q, hint)),
        call(A[2], CSYS, cusr(q)))
    chains = [x[0] for x in r0]; ans = [x[1] for x in r0]   # [misled,misled,clean]

    if mode == "FM-3.2":             # no verification → emit majority
        fin = Counter([a for a in ans if a is not None]).most_common(1)[0][0]
        return fin == truth, bud.current_cost

    rec = MathSolverAgent(solver_index=8, name="Rec", tier=ModelTier.SLOW,
        llm=slow, blackboard=bb, metrics=m, budget_monitor=bud)
    rp = MathReconcilerPrompts()
    if mode == "FM-2.5":             # ignore peer: drop the clean chain
        idx = [0, 1]
    elif mode == "FM-1.3":           # step repetition: misled chain ×3
        chains = [chains[0], chains[0], chains[0]]; ans = [ans[0]]*3; idx = [0, 1, 2]
    else:                            # none / FM-2.6 → full context
        idx = [0, 1, 2]
    nm = ["S1", "S2", "S3"]
    summary = ", ".join(f"{nm[i]}={ans[i]}" for i in idx)
    um = rp.format_user(q, summary, [(nm[i], chains[i]) for i in idx])
    rr, fin = await call(rec, rp.SYSTEM, um)
    if mode == "FM-2.6":             # reasoning-action mismatch: clobber answer
        fin = tgt
    return fin == truth, bud.current_cost


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for mode in ("none", "FM-2.5", "FM-3.2", "FM-1.3", "FM-2.6"):
        npass = 0; cost = 0.0
        for p in probs:
            bud = BudgetMonitor(max_cost_per_problem=0.6, max_iterations=20,
                                max_llm_calls=40)
            try:
                ok, c = await asyncio.wait_for(
                    episode(p, mode, fast, slow, bud), timeout=150)
            except Exception:
                ok, c = False, bud.current_cost
            npass += int(ok); cost += c
            print(f"[{mode:7s}] {p['problem_id']:22s} {'PASS' if ok else 'FAIL'}")
        out[mode] = {"summary": {"injected_mode": mode, "n": len(probs),
            "accuracy": npass/len(probs), "total_cost_usd": cost}}
        print(f"== {mode}: acc={out[mode]['summary']['accuracy']:.0%} "
              f"cost=${cost:.3f} ==")
    base = out["none"]["summary"]["accuracy"]
    for mode in out:
        out[mode]["summary"]["containment_drop"] = round(
            base - out[mode]["summary"]["accuracy"], 3)
    json.dump(out, open(RES/f"w9_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved w9_{ts}.json (baseline acc={base:.0%})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
