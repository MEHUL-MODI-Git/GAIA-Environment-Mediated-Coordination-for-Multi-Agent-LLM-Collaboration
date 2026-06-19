#!/usr/bin/env python3
"""NX5b — Branch-and-Merge operating regime (when does Feature F help?).

Companion to NX5. NX5 used a STRONG solver with no headroom (pass@1=89%) and
found branch-and-merge gives ~0 benefit. NX5b varies SOLVER STRENGTH so we can
show the benefit appears exactly when there is recoverable headroom — the
regime predicted by repeated-sampling coverage scaling (Large Language Monkeys,
2407.21787) with an oracle (unit-test) selector.

One knob: --model sets the WHOLE agent pool to that model (uniform strength).
  WEAK   : gpt-4.1-nano   (expect headroom -> branch_on >> branch_off)
  STRONG : gpt-4.1-mini   (NX5-style -> ~0 benefit)

Slice: probs[start::step][:n]  (default a spread, representative slice).

Usage:
  python run_nx5b.py --model gpt-4.1-nano --n 8 --start 3 --step 5
"""
import argparse, asyncio, json, os, sys, time, logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
_env = ROOT / ".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy
from gaia.agents.coder import CoderAgent
from gaia.agents.critic import CriticAgent
from gaia.agents.verifier import VerifierAgent
from gaia.agents.edge_case import EdgeCaseAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.loop import EpisodeLoop
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT / "data" / "humaneval" / "test.jsonl"
HERE = Path(__file__).parent.parent
RES, LOGS = HERE / "results", HERE / "logs"


def load(n, start, step):
    probs = [json.loads(l) for l in open(DATA) if l.strip()]
    return probs[start::step][:n]


async def run_one(problem, branch_on, llm, n_branches):
    tid = problem.get("task_id", "?"); safe = tid.replace("/", "_")
    tag = f"{safe}_{'on' if branch_on else 'off'}"
    bb = Blackboard(log_file=LOGS / f"{tag}.jsonl")
    bud = BudgetMonitor(max_cost_per_problem=0.8, max_iterations=12, max_llm_calls=40)
    pol = Policy(max_iterations=8, branch_trigger_on_failure=branch_on,
                 branch_max_parallel=n_branches,
                 verification_strictness="all_tests_pass", stop_on_first_pass=True)
    # Uniform-strength pool: every agent uses the same model.
    agents = [
        CoderAgent(name="Coder-1", tier=ModelTier.FAST, llm=llm, blackboard=bb, budget_monitor=bud),
        CoderAgent(name="Coder-2", tier=ModelTier.FAST, llm=llm, blackboard=bb, budget_monitor=bud),
        CriticAgent(name="Critic-1", tier=ModelTier.FAST, llm=llm, blackboard=bb, budget_monitor=bud),
        VerifierAgent(name="Verifier-1", tier=ModelTier.FAST, llm=llm, blackboard=bb, budget_monitor=bud),
        EdgeCaseAgent(name="EdgeCase-1", tier=ModelTier.FAST, llm=llm, blackboard=bb, budget_monitor=bud),
    ]
    loop = EpisodeLoop(blackboard=bb, agents=agents, metrics=MetricsCollector(),
                       policy=pol, budget_monitor=bud)
    t0 = time.time()
    try:
        res = await loop.run_episode(problem)
        ok = bool(res.passed); nbr = getattr(res, "branches_created", 0) or 0
    except Exception as e:
        print(f"   ! {tid} error: {e}"); ok, nbr = False, 0
    return ok, nbr, bud.current_cost, round(time.time() - t0, 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-nano")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=3)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n-branches", type=int, default=3)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    probs = load(args.n, args.start, args.step)
    llm = OpenAILLM(model=args.model, tier=ModelTier.FAST)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"NX5b  model={args.model}  n={len(probs)}  slice=[{args.start}::{args.step}]  n_branches={args.n_branches}")

    out = {"meta": {"model": args.model, "n": len(probs), "start": args.start,
                    "step": args.step, "n_branches": args.n_branches, "ts": ts}}
    for cond, bon in [("branch_off", False), ("branch_on", True)]:
        npass = nbr = 0; cost = 0.0; recs = []
        for p in probs:
            ok, b, c, dur = await run_one(p, bon, llm, args.n_branches)
            npass += int(ok); nbr += b; cost += c
            recs.append({"task_id": p.get("task_id"), "passed": ok, "branches": b,
                         "cost": round(c, 4), "dur_s": dur})
            print(f"[{cond:10s}] {p.get('task_id'):16s} {'PASS' if ok else 'FAIL'} br={b}")
        out[cond] = {"summary": {"condition": cond, "n": len(probs),
                     "accuracy": npass / len(probs), "total_branches": nbr,
                     "total_cost_usd": round(cost, 4)}, "results": recs}
        s = out[cond]["summary"]
        print(f"== {cond}: acc={s['accuracy']:.0%} branches={nbr} cost=${cost:.3f} ==")

    off = {r["task_id"]: r["passed"] for r in out["branch_off"]["results"]}
    on = {r["task_id"]: r["passed"] for r in out["branch_on"]["results"]}
    rec = [t for t in off if not off[t] and on.get(t)]
    reg = [t for t in off if off[t] and not on.get(t)]
    out["_recovery"] = {"recovered": rec, "regressed": reg,
                        "n_recovered": len(rec), "n_regressed": len(reg)}
    print(f"\n>>> model={args.model}: recovered={len(rec)} regressed={len(reg)} "
          f"(Δacc={out['branch_on']['summary']['accuracy']-out['branch_off']['summary']['accuracy']:+.1%})")
    fn = RES / f"nx5b_{args.model.replace('.','_')}_{ts}.json"
    json.dump(out, open(fn, "w"), indent=2, default=str)
    print(f"Saved {fn.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
