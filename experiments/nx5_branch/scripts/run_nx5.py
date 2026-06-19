#!/usr/bin/env python3
"""NX5 — Branch-and-Merge value isolation (Feature F).

Feature F (fork the blackboard on stall → run N diverse parallel attempts →
merge the winner) was implemented + logging-fixed but never isolated as an
experiment. Here we run the SAME HumanEval problems through the standard
EpisodeLoop with Feature F OFF vs ON and measure how many otherwise-failing
problems it recovers, at what extra cost.

branch_off : policy.branch_trigger_on_failure = False (single trajectory)
branch_on  : policy.branch_trigger_on_failure = True, n_branches=3

We deliberately use a HARDER slice (every 4th problem from 40..) to surface
stalls so the mechanism is actually exercised.
"""
import asyncio, json, os, sys, time, logging
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

DATA = ROOT/"data"/"humaneval"/"test.jsonl"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"logs"
FAST, SLOW = "gpt-4.1-mini", "gpt-4.1"


def load(n):
    probs = [json.loads(l) for l in open(DATA) if l.strip()]
    # harder slice: spread across the set, skip the trivial early ones
    return probs[20::6][:n]


async def run_one(problem, branch_on, fast, slow):
    tid = problem.get("task_id", "?")
    safe = tid.replace("/", "_")
    bb = Blackboard(log_file=LOGS/f"{safe}_{'on' if branch_on else 'off'}.jsonl")
    bud = BudgetMonitor(max_cost_per_problem=0.8, max_iterations=12, max_llm_calls=40)
    pol = Policy(max_iterations=8, branch_trigger_on_failure=branch_on,
                 branch_max_parallel=3, verification_strictness="all_tests_pass",
                 stop_on_first_pass=True)
    agents = [
        CoderAgent(name="Coder-1", tier=ModelTier.FAST, llm=fast,
                   blackboard=bb, budget_monitor=bud),
        CoderAgent(name="Coder-2", tier=ModelTier.FAST, llm=fast,
                   blackboard=bb, budget_monitor=bud),
        CriticAgent(name="Critic-1", tier=ModelTier.FAST, llm=fast,
                    blackboard=bb, budget_monitor=bud),
        VerifierAgent(name="Verifier-1", tier=ModelTier.FAST, llm=fast,
                      blackboard=bb, budget_monitor=bud),
        EdgeCaseAgent(name="EdgeCase-1", tier=ModelTier.SLOW, llm=slow,
                      blackboard=bb, budget_monitor=bud),
    ]
    loop = EpisodeLoop(blackboard=bb, agents=agents, metrics=MetricsCollector(),
                       policy=pol, budget_monitor=bud)
    t0 = time.time()
    try:
        res = await loop.run_episode(problem)
        ok = bool(res.passed)
        nbr = getattr(res, "branches_created", 0) or 0
    except Exception as e:
        ok, nbr = False, 0
    return ok, nbr, bud.current_cost, round(time.time()-t0, 1)


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    probs = load(n)
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for cond, bon in [("branch_off", False), ("branch_on", True)]:
        npass = nbr = 0; cost = 0.0; recs = []
        for p in probs:
            ok, b, c, dur = await run_one(p, bon, fast, slow)
            npass += int(ok); nbr += b; cost += c
            recs.append({"task_id": p.get("task_id"), "passed": ok,
                         "branches": b, "cost": c, "dur_s": dur})
            print(f"[{cond:10s}] {p.get('task_id'):16s} "
                  f"{'PASS' if ok else 'FAIL'} br={b}")
        out[cond] = {"summary": {"condition": cond, "n": len(probs),
            "accuracy": npass/len(probs), "total_branches": nbr,
            "total_cost_usd": cost}, "results": recs}
        s = out[cond]["summary"]
        print(f"== {cond}: acc={s['accuracy']:.0%} branches={nbr} "
              f"cost=${cost:.3f} ==")
    # recovery analysis: problems off-FAILS that on-PASSES
    off = {r["task_id"]: r["passed"] for r in out["branch_off"]["results"]}
    on = {r["task_id"]: r["passed"] for r in out["branch_on"]["results"]}
    recovered = [t for t in off if not off[t] and on.get(t)]
    regressed = [t for t in off if off[t] and not on.get(t)]
    out["_recovery"] = {"recovered_by_branching": recovered,
                        "regressed": regressed}
    print(f"\nRecovered by Feature F: {len(recovered)}  Regressed: {len(regressed)}")
    json.dump(out, open(RES/f"nx5_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved nx5_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
