#!/usr/bin/env python3
"""NX1 — Real-framework baseline + plain-blackboard ablation (E3 trap suite).

Closes the most critical gap: GAIA is only compared to single/majority/its own
ablations. NX1 adds two principled external baselines on the SAME 13 validated
trap problems, so we can attribute GAIA's win to *structured conflict-as-task*
rather than "having a blackboard" or "doing debate".

Conditions added (compared against existing E3: single=100, majority=15.4,
gaia=100):

  debate            — AutoGen/society-of-minds style. 3 solvers answer
                       independently, then 2 rounds where each sees the others'
                       full answers+reasoning and may revise. Final = majority.
                       (No shared board, no typed signals, no verifier.)

  blackboard_plain  — the 2025 LLM-blackboard design (arXiv 2510.01285 style):
                       agents post answers+reasoning to a shared buffer all can
                       read, a synthesizer reads the whole board and emits a
                       final answer. NO CONFLICT signal, NO conflict-as-task,
                       NO reconciler trigger. Isolates "shared buffer alone".

The contrast debate vs blackboard_plain vs GAIA tells us whether the active
ingredient is (a) iterative mutual revision, (b) a shared buffer, or
(c) GAIA's structured conflict-as-task escalation.

Usage: python experiments/nx1_baselines/scripts/run_nx1.py
"""
import asyncio, json, os, sys, time, logging
from collections import Counter
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
from gaia.agents.math import MathSolverAgent
from gaia.agents.math.math_solver import extract_final_answer
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
from gaia.utils.checkpoint import CheckpointManager

DATA = ROOT / "data" / "gsm8k" / "correlated_failure_problems.json"
RES = Path(__file__).parent.parent / "results"
LOGS = Path(__file__).parent.parent / "logs"
FAST = "gpt-4.1-nano"

SYS = MathSolverPrompts.SYSTEM
def uprompt(q): return MathSolverPrompts.format_user(q)
MSYS = MisledSolverPrompts.SYSTEM
def musr(q, h): return MisledSolverPrompts.format_user(q, h)

# CRITICAL: NX1 must face the SAME correlated-failure stressor as E3's
# gaia/majority conditions — 2 MISLED agents (shared wrong heuristic) + 1 CLEAN.
# Otherwise we'd just re-measure single-agent accuracy. agent_kind[i]:
#   0,1 -> misled (get the misleading_hint)   2 -> clean
def _msgs_for(kind, q, hint, extra=""):
    if kind == "clean":
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": uprompt(q) + extra}]
    return [{"role": "system", "content": MSYS},
            {"role": "user", "content": musr(q, hint) + extra}]


async def cond_debate(problem, llm, budget):
    """2 misled + 1 clean; 2 rounds of mutual revision; majority vote.
    Tests: does iterative debate let the lone correct agent overturn the
    misled majority, or does social pressure entrench the shared error?"""
    q, truth, hint = problem["question"], problem["answer"], problem["misleading_hint"]
    bb = Blackboard(log_file=LOGS / f"{problem['problem_id']}_debate.jsonl")
    kinds = ["misled", "misled", "clean"]
    agents = [MathSolverAgent(solver_index=i, name=f"Deb-{i}", tier=ModelTier.FAST,
                              llm=llm, blackboard=bb, metrics=MetricsCollector(),
                              budget_monitor=budget) for i in range(3)]
    async def solve(i, a, extra=""):
        r = await a.call_llm(_msgs_for(kinds[i], q, hint, extra), temperature=0.0)
        return r, extract_final_answer(r)
    resp = await asyncio.gather(*[solve(i, a) for i, a in enumerate(agents)])
    chains = [r[0] for r in resp]; ans = [r[1] for r in resp]
    for _ in range(2):
        async def revise(i, a):
            others = "\n\n".join(f"[Peer {j}] {chains[j][:600]}"
                                 for j in range(3) if j != i)
            extra = (f"\n\nPeers proposed solutions below. Reconsider and give "
                     f"your best final answer.\n{others}")
            return await solve(i, a, extra)
        resp = await asyncio.gather(*[revise(i, a) for i, a in enumerate(agents)])
        chains = [r[0] for r in resp]; ans = [r[1] for r in resp]
    valid = [x for x in ans if x is not None]
    final = Counter(valid).most_common(1)[0][0] if valid else None
    return final, (final == truth), budget.current_cost, {"round_answers": ans}


async def cond_bb_plain(problem, llm, budget):
    """Plain shared buffer (the 2025 LLM-blackboard design): 2 misled + 1 clean
    post to a board, a synthesizer reads ALL and decides. NO CONFLICT signal,
    NO conflict-as-task, NO reconciler. Isolates 'shared buffer alone' under
    the same stressor GAIA faces."""
    q, truth, hint = problem["question"], problem["answer"], problem["misleading_hint"]
    bb = Blackboard(log_file=LOGS / f"{problem['problem_id']}_bbplain.jsonl")
    kinds = ["misled", "misled", "clean"]
    solvers = [MathSolverAgent(solver_index=i, name=f"BB-{i}", tier=ModelTier.FAST,
                               llm=llm, blackboard=bb, metrics=MetricsCollector(),
                               budget_monitor=budget) for i in range(3)]
    async def solve(i, a):
        r = await a.call_llm(_msgs_for(kinds[i], q, hint), temperature=0.0)
        return r
    chains = await asyncio.gather(*[solve(i, a) for i, a in enumerate(solvers)])
    board = "\n\n".join(f"[Agent {i} posted]\n{c[:800]}" for i, c in enumerate(chains))
    synth = MathSolverAgent(solver_index=9, name="BB-Synth", tier=ModelTier.FAST,
                            llm=llm, blackboard=bb, metrics=MetricsCollector(),
                            budget_monitor=budget)
    smsg = [{"role": "system", "content":
             "You read a shared blackboard of independent solver write-ups and "
             "must output the single best final integer answer. End with "
             "**Final Answer: [integer]**."},
            {"role": "user", "content":
             f"Problem: {q}\n\nShared blackboard:\n{board}\n\n"
             f"Give the best final answer. End with **Final Answer: [integer]**"}]
    sresp = await synth.call_llm(smsg, temperature=0.0)
    final = extract_final_answer(sresp)
    return final, (final == truth), budget.current_cost, {}


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    problems = json.load(open(DATA))
    llm = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = CheckpointManager(RES / f"nx1_checkpoint_{ts}.json")
    done = set(ckpt.get_completed_task_ids())
    out = {"debate": {"results": []}, "blackboard_plain": {"results": []}}

    for cond, fn in [("debate", cond_debate), ("blackboard_plain", cond_bb_plain)]:
        n_pass = 0
        for i, p in enumerate(problems, 1):
            key = f"{cond}/{p['problem_id']}"
            if key in done:
                continue
            budget = BudgetMonitor(max_cost_per_problem=1.0, max_iterations=30,
                                   max_llm_calls=60)
            t0 = time.time()
            try:
                final, ok, cost, extra = await fn(p, llm, budget)
            except Exception as e:
                final, ok, cost, extra = None, False, budget.current_cost, {"error": str(e)}
            n_pass += int(ok)
            rec = {"problem_id": p["problem_id"], "condition": cond,
                   "passed": ok, "proposed_answer": final,
                   "ground_truth": p["answer"], "trap_category": p["category"],
                   "cost_usd": cost, "duration_s": round(time.time() - t0, 2),
                   **extra}
            out[cond]["results"].append(rec)
            ckpt.add_result(task_id=key, passed=ok, iterations=1, cost_usd=cost,
                            duration_s=rec["duration_s"],
                            stop_reason="passed" if ok else "failed")
            print(f"[{cond}] {i:2d}/{len(problems)} {p['problem_id']:22s} "
                  f"{'PASS' if ok else 'FAIL'} ans={final} truth={p['answer']}")
        rs = out[cond]["results"]
        acc = sum(r["passed"] for r in rs) / len(rs) if rs else 0
        out[cond]["summary"] = {"condition": cond, "n": len(rs),
                                "accuracy": acc,
                                "total_cost_usd": sum(r["cost_usd"] for r in rs)}
        print(f"\n== {cond}: acc={acc:.1%} cost=${out[cond]['summary']['total_cost_usd']:.4f} ==\n")

    op = RES / f"nx1_{ts}.json"
    json.dump(out, open(op, "w"), indent=2, default=str)
    print(f"Saved {op}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
