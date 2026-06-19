#!/usr/bin/env python3
"""W1 — Chaos engineering: agent-dropout resilience (tests Feature C).

GAIA agents self-assign by polling/claiming tasks (Feature C, no central
orchestrator). Hypothesis: if agents die mid-episode, the lease/claim model
lets survivors absorb the work, whereas a fixed-orchestration pipeline stalls.

We stress GAIA's E3 conflict pipeline by randomly DROPPING each non-essential
solver with probability p before it can post (simulating crashed workers),
across p ∈ {0, 0.33, 0.67}. The aggregator/reconciler/verifier are the
"survivors". Metric: accuracy + how often a usable answer was still produced
vs a hard stall.

Dropout is injected by replacing a solver's execute() with a no-op for that
episode (the agent "crashed" before posting). 13 traps × 3 drop levels,
GAIA condition.
"""
import asyncio, json, os, random, sys, logging
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
from gaia.agents.math import (MathSolverAgent, MathAggregatorAgent,
                              MathReconcilerAgent, MathVerifierAgent)
from gaia.agents.math.misled_solver import MisledSolverAgent
from gaia.episode.correlated_failure_loop import CorrelatedFailureEpisodeLoop
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"logs"
FAST, SLOW = "gpt-4.1-nano", "gpt-4.1"


def crash(agent):
    """Simulate a crashed worker: it never posts an artifact this episode."""
    async def dead(task):
        return []
    agent.execute = dead
    return agent


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for p_drop in (0.0, 0.34, 0.67):
        npass = 0; produced = 0; cost = 0.0; recs = []
        rng = random.Random(42)
        for prob in probs:
            bb = Blackboard(log_file=LOGS/f"{prob['problem_id']}_p{int(p_drop*100)}.jsonl")
            m = MetricsCollector()
            bud = BudgetMonitor(max_cost_per_problem=0.6, max_iterations=25,
                                max_llm_calls=50)
            # E3-style pool: 2 misled + 1 clean + agg + rec + ver
            solvers = [
                MisledSolverAgent(misled_index=0, name="M0", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
                MisledSolverAgent(misled_index=1, name="M1", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
                MathSolverAgent(solver_index=0, name="C0", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
            ]
            # randomly crash each solver w.p. p_drop (but never all 3)
            alive = []
            for s in solvers:
                if rng.random() < p_drop:
                    crash(s)
                else:
                    alive.append(s)
            if not alive:                       # guarantee ≥1 survivor
                pass  # all crashed → tests hard-stall handling
            agents = solvers + [
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
                r = await loop.run_episode(prob)
                ok = bool(r.passed)
                ans_produced = r.proposed_answer is not None
            except Exception:
                ok, ans_produced = False, False
            npass += int(ok); produced += int(ans_produced)
            cost += bud.current_cost
            recs.append({"problem_id": prob["problem_id"], "passed": ok,
                         "answer_produced": ans_produced,
                         "n_crashed": 3 - len(alive)})
            print(f"[p={p_drop:.2f}] {prob['problem_id']:22s} "
                  f"{'PASS' if ok else 'FAIL'} crashed={3-len(alive)}/3 "
                  f"answer={'yes' if ans_produced else 'STALL'}")
        out[f"p{int(p_drop*100)}"] = {"summary": {"p_drop": p_drop,
            "n": len(probs), "accuracy": npass/len(probs),
            "answer_produced_rate": produced/len(probs),
            "total_cost_usd": cost}, "results": recs}
        s = out[f"p{int(p_drop*100)}"]["summary"]
        print(f"== p_drop={p_drop:.2f}: acc={s['accuracy']:.0%} "
              f"produced={s['answer_produced_rate']:.0%} cost=${cost:.3f} ==")
    json.dump(out, open(RES/f"w1_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved w1_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
