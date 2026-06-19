#!/usr/bin/env python3
"""W3 — Emergent specialization without assigned roles.

E8 showed identical agents WITHOUT a blackboard (independent majority vote)
are flat. W3 asks the complementary question: if N *identical* agents share
ONE blackboard and an open prompt (no roles assigned), do roles
SPONTANEOUSLY emerge from the medium?

4 identical agents, same generic prompt: "read the shared board; contribute
the single most useful next action — SOLVE, VERIFY, CRITIQUE, or SYNTHESIZE —
and label it." 4 rounds on the 13 E3 traps. We tag each contribution by the
self-declared action, then per agent compute the action distribution and its
entropy across rounds. Emergence ⇔ agents' action distributions DIVERGE (each
converges to a distinct dominant action) rather than all staying uniform.

Metric: mean per-agent action-entropy (low = specialized) and pairwise
distribution divergence (high = differentiated roles). Honest: this measures
behavioural differentiation, not a causal claim that it improves accuracy.
"""
import asyncio, json, os, re, sys, math, logging
from collections import Counter, defaultdict
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
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = Path(__file__).parent.parent/"results"
FAST = "gpt-4.1-nano"
N_AGENTS, N_ROUNDS = 4, 4

GEN_SYS = ("You are ONE of several identical agents collaborating on a shared "
           "scratchpad to solve a problem. No roles are assigned. Read the "
           "scratchpad, then contribute the SINGLE most useful next action. "
           "Begin your reply with exactly one tag: [SOLVE] (work the problem), "
           "[VERIFY] (check an existing answer's arithmetic), [CRITIQUE] "
           "(point out a flaw in existing work), or [SYNTHESIZE] (combine "
           "existing work into a final answer). Then do that action. If you "
           "produce a final number end with **Final Answer: [int]**.")


def tag_of(text):
    m = re.match(r"\s*\[(SOLVE|VERIFY|CRITIQUE|SYNTHESIZE)\]", text or "",
                 re.IGNORECASE)
    return m.group(1).upper() if m else "SOLVE"


def entropy(counts):
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    return -sum((c/tot)*math.log2(c/tot) for c in counts.values() if c)


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    llm = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # action distribution per agent, aggregated over all problems/rounds
    agent_actions = defaultdict(Counter)
    correct = 0
    for p in probs:
        q, truth = p["question"], p["answer"]
        bb = Blackboard(log_file=Path("/tmp/_w3.jsonl"))
        agents = [MathSolverAgent(solver_index=i, name=f"Ag{i}",
            tier=ModelTier.FAST, llm=llm, blackboard=bb,
            metrics=MetricsCollector(),
            budget_monitor=BudgetMonitor(max_cost_per_problem=0.6,
                max_iterations=20, max_llm_calls=40)) for i in range(N_AGENTS)]
        board = []
        last_ans = None
        for rnd in range(N_ROUNDS):
            async def act(i):
                bt = "\n\n".join(board[-6:]) or "(empty)"
                u = (f"Problem: {q}\n\nShared scratchpad so far:\n{bt}\n\n"
                     f"Your single most useful action now:")
                r = await agents[i].call_llm(
                    [{"role": "system", "content": GEN_SYS},
                     {"role": "user", "content": u}], temperature=0.4)
                return i, r
            results = await asyncio.gather(*[act(i) for i in range(N_AGENTS)])
            for i, r in results:
                tg = tag_of(r)
                agent_actions[i][tg] += 1
                board.append(f"[Ag{i}|{tg}] {r[:280]}")
                a = extract_final_answer(r)
                if a is not None:
                    last_ans = a
        if last_ans == truth:
            correct += 1
        print(f"{p['problem_id']:22s} final={last_ans} truth={truth} "
              f"{'OK' if last_ans==truth else '--'}")

    # specialization metrics
    per_agent_entropy = {i: round(entropy(agent_actions[i]), 3)
                         for i in range(N_AGENTS)}
    dominant = {i: (agent_actions[i].most_common(1)[0][0]
                    if agent_actions[i] else None) for i in range(N_AGENTS)}
    n_distinct_dominant = len(set(d for d in dominant.values() if d))
    mean_H = round(sum(per_agent_entropy.values())/N_AGENTS, 3)
    maxH = math.log2(4)  # 4 action types
    out = {"summary": {
        "n_agents": N_AGENTS, "n_rounds": N_ROUNDS, "n_problems": len(probs),
        "task_accuracy": round(correct/len(probs), 3),
        "per_agent_action_counts": {i: dict(agent_actions[i])
                                    for i in range(N_AGENTS)},
        "per_agent_entropy_bits": per_agent_entropy,
        "mean_agent_entropy_bits": mean_H,
        "max_possible_entropy_bits": round(maxH, 3),
        "dominant_action_per_agent": dominant,
        "n_distinct_dominant_roles": n_distinct_dominant,
        "emergence": (n_distinct_dominant >= 3 and mean_H < 0.85*maxH)}}
    json.dump(out, open(RES/f"w3_{ts}.json", "w"), indent=2, default=str)
    s = out["summary"]
    print(f"\n== W3 ==")
    print(f"per-agent action counts: {s['per_agent_action_counts']}")
    print(f"dominant role/agent: {s['dominant_action_per_agent']}")
    print(f"distinct dominant roles: {s['n_distinct_dominant_roles']}/4  "
          f"mean entropy: {mean_H}/{round(maxH,2)} bits  "
          f"emergence={'YES' if s['emergence'] else 'NO'}")
    print(f"Saved w3_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
