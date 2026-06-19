#!/usr/bin/env python3
"""NX9-dose — GAIA robustness vs correlated-corruption dose.

Sweeps k = #misled agents in {0,1,2,3} (pool = k misled + (3-k) clean) through
GAIA's full conflict-as-task pipeline on the 13-trap suite. Produces a
dose-response curve: at what corruption fraction does GAIA's reconciler stop
rescuing the answer? Novel: a robustness curve no baseline framework can match
because only GAIA has the conflict-as-task escalation being stressed.
"""
import asyncio, glob, json, os, sys, logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
_env = ROOT/".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        ln=ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k,v=ln.split("=",1); os.environ.setdefault(k.strip(),v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy
from gaia.agents.math import MathAggregatorAgent, MathReconcilerAgent, MathVerifierAgent, MathSolverAgent
from gaia.agents.math.misled_solver import MisledSolverAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.correlated_failure_loop import CorrelatedFailureEpisodeLoop
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"logs"/"dose"
FAST, SLOW = "gpt-4.1-nano", "gpt-4.1"


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for k in (0, 1, 2, 3):                       # misled dose
        npass = 0; cost = 0.0; recs = []
        for p in probs:
            bb = Blackboard(log_file=LOGS/f"{p['problem_id']}_k{k}.jsonl")
            m = MetricsCollector()
            bud = BudgetMonitor(max_cost_per_problem=1.0, max_iterations=25, max_llm_calls=50)
            agents = []
            for i in range(k):
                agents.append(MisledSolverAgent(misled_index=i, name=f"M{i}",
                    tier=ModelTier.FAST, llm=fast, blackboard=bb, metrics=m, budget_monitor=bud))
            for j in range(3 - k):
                agents.append(MathSolverAgent(solver_index=j, name=f"C{j}",
                    tier=ModelTier.FAST, llm=fast, blackboard=bb, metrics=m, budget_monitor=bud))
            agents += [
                MathAggregatorAgent(name="Agg", tier=ModelTier.FAST, llm=fast,
                    blackboard=bb, metrics=m, budget_monitor=bud),
                MathReconcilerAgent(name="Rec", tier=ModelTier.SLOW, llm=slow,
                    blackboard=bb, metrics=m, budget_monitor=bud),
                MathVerifierAgent(name="Ver", tier=ModelTier.FAST, llm=fast,
                    blackboard=bb, metrics=m, budget_monitor=bud),
            ]
            loop = CorrelatedFailureEpisodeLoop(blackboard=bb, agents=agents,
                metrics=m, policy=Policy(), budget_monitor=bud)
            try:
                r = await loop.run_episode(p)
                ok = r.passed; cost += r.cost_usd
                recs.append({"problem_id": p["problem_id"], "passed": ok,
                             "conflict_detected": r.conflict_detected})
            except Exception as e:
                ok = False; recs.append({"problem_id": p["problem_id"],
                                         "passed": False, "error": str(e)})
            npass += int(ok)
        acc = npass/len(probs)
        cr = sum(1 for r in recs if r.get("conflict_detected"))/len(probs)
        out[f"k{k}"] = {"summary": {"misled": k, "clean": 3-k, "n": len(probs),
                        "accuracy": acc, "conflict_rate": cr,
                        "total_cost_usd": cost}, "results": recs}
        print(f"misled={k} clean={3-k}  acc={acc:.1%}  conflict_rate={cr:.0%}  cost=${cost:.3f}")
    json.dump(out, open(RES/f"dose_{ts}.json", "w"), indent=2)
    print(f"Saved {RES/f'dose_{ts}.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
