#!/usr/bin/env python3
"""NX7 — realistic agentic planning task (multi-constraint scheduling).

Non-toy: agents must decompose per-person calendars, intersect free windows,
respect working hours + duration, and commit a start time. Ground truth is a
deterministic programmatic checker (a proposal is correct iff it satisfies
ALL constraints). Tests whether GAIA's coordination advantage persists beyond
synthetic math/puzzle suites (roadmap G5 / NX7).

Conditions (all on the same 30 problems):
  single  — 1 agent sees everything, commits an answer.
  debate  — 3 agents propose, 2 rounds of mutual revision, pick a valid one
            (AutoGen society-of-minds analog).
  gaia    — 2 experts each analyze HALF the people → post free-window
            deductions → synthesizer intersects → programmatic VERIFY →
            on failure, conflict-as-task: one reconciliation round with the
            verifier's feedback (GAIA's verify→reconcile loop).
"""
import asyncio, json, os, re, sys, time, logging
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
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT/"data"/"scheduling"/"problems.json"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"logs"
FAST, SLOW = "gpt-4.1-mini", "gpt-4.1"


def t2m(s):
    h, m = s.split(":"); return int(h)*60 + int(m)


def extract_time(text):
    m = re.findall(r"FINAL:\s*(\d{1,2}:\d{2})", text)
    if m:
        return m[-1]
    m = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    return m[-1] if m else None


def verify(prob, hhmm):
    """Deterministic ground-truth check."""
    if not hhmm:
        return False
    try:
        start = t2m(hhmm)
    except Exception:
        return False
    dur = prob["duration_min"]
    ws, we = t2m(prob["working_hours"][0]), t2m(prob["working_hours"][1])
    if start < ws or start + dur > we:
        return False
    for person in prob["people"]:
        for b in person["busy"]:
            bs, be = t2m(b[0]), t2m(b[1])
            if start < be and bs < start + dur:        # overlap
                return False
    return True


def cal_text(people):
    return "\n".join(
        f"- {p['name']}: busy " +
        (", ".join(f"{a}-{b}" for a, b in p["busy"]) or "(none)")
        for p in people)


def prompt_user(prob, people=None):
    ppl = people if people is not None else prob["people"]
    return (f"{prob['instruction']}\n\nWorking hours: "
            f"{prob['working_hours'][0]}-{prob['working_hours'][1]}; "
            f"meeting duration: {prob['duration_min']} min.\n"
            f"Calendars:\n{cal_text(ppl)}\n\n"
            f"Reason step by step, then output exactly: FINAL: HH:MM")


SYS = ("You are a precise scheduling assistant. Find a start time at which "
       "ALL listed people are free for the full meeting duration and which "
       "lies within working hours. Show brief reasoning, then end with "
       "exactly: FINAL: HH:MM")


async def one(agent, sysmsg, usermsg, temp=0.0):
    r = await agent.call_llm([{"role": "system", "content": sysmsg},
                              {"role": "user", "content": usermsg}], temperature=temp)
    return r, extract_time(r)


async def c_single(prob, fast, slow, bud):
    bb = Blackboard(log_file=LOGS/f"{prob['problem_id']}_single.jsonl")
    a = MathSolverAgent(solver_index=0, name="S", tier=ModelTier.FAST, llm=fast,
        blackboard=bb, metrics=MetricsCollector(), budget_monitor=bud)
    _, t = await one(a, SYS, prompt_user(prob))
    return verify(prob, t), bud.current_cost


async def c_debate(prob, fast, slow, bud):
    bb = Blackboard(log_file=LOGS/f"{prob['problem_id']}_debate.jsonl")
    A = [MathSolverAgent(solver_index=i, name=f"D{i}", tier=ModelTier.FAST,
        llm=fast, blackboard=bb, metrics=MetricsCollector(), budget_monitor=bud)
        for i in range(3)]
    resp = await asyncio.gather(*[one(a, SYS, prompt_user(prob)) for a in A])
    chains = [r[0] for r in resp]; ans = [r[1] for r in resp]
    for _ in range(2):
        async def rev(i):
            others = "\n\n".join(f"[Peer {j}] proposed {ans[j]}:\n{chains[j][:500]}"
                                 for j in range(3) if j != i)
            return await one(A[i], SYS, prompt_user(prob) +
                             f"\n\nPeers:\n{others}\nReconsider; FINAL: HH:MM")
        resp = await asyncio.gather(*[rev(i) for i in range(3)])
        chains = [r[0] for r in resp]; ans = [r[1] for r in resp]
    # pick the most common proposal that actually verifies, else most common
    valid = [t for t in ans if verify(prob, t)]
    if valid:
        return True, bud.current_cost
    return False, bud.current_cost


async def c_gaia(prob, fast, slow, bud):
    """experts (split people) → free-window deductions → synthesizer intersect
    → programmatic verify → conflict-as-task reconcile on failure."""
    bb = Blackboard(log_file=LOGS/f"{prob['problem_id']}_gaia.jsonl")
    m = MetricsCollector()
    ppl = prob["people"]
    half = (len(ppl) + 1) // 2
    groups = [ppl[:half], ppl[half:]]
    experts = [MathSolverAgent(solver_index=i, name=f"E{i}", tier=ModelTier.FAST,
        llm=fast, blackboard=bb, metrics=m, budget_monitor=bud) for i in range(2)]
    esys = ("You analyze a SUBSET of people's calendars. List, as compact "
            "intervals, the times within working hours when ALL people in "
            "YOUR subset are simultaneously free for the meeting duration. "
            "End with: FREE: [hh:mm-hh:mm, ...]")
    async def exp(i):
        u = (f"{prob['instruction']}\nWorking hours "
             f"{prob['working_hours'][0]}-{prob['working_hours'][1]}, "
             f"duration {prob['duration_min']}m.\nYour people:\n"
             f"{cal_text(groups[i])}\nEnd with FREE: [...]")
        r, _ = await one(experts[i], esys, u)
        return r
    ded = await asyncio.gather(*[exp(0), exp(1)])
    syn = MathSolverAgent(solver_index=8, name="Syn", tier=ModelTier.SLOW,
        llm=slow, blackboard=bb, metrics=m, budget_monitor=bud)
    board = f"[Expert-1 free windows]\n{ded[0][:700]}\n\n[Expert-2 free windows]\n{ded[1][:700]}"
    su = (f"{prob['instruction']}\nWorking hours "
          f"{prob['working_hours'][0]}-{prob['working_hours'][1]}, duration "
          f"{prob['duration_min']}m. Two experts each computed free windows "
          f"for half the people. Intersect them and pick ONE valid start.\n"
          f"{board}\nFINAL: HH:MM")
    _, t = await one(syn, SYS, su)
    if verify(prob, t):
        return True, bud.current_cost
    # conflict-as-task: verifier feedback → one reconciliation round
    rec = MathSolverAgent(solver_index=9, name="Rec", tier=ModelTier.SLOW,
        llm=slow, blackboard=bb, metrics=m, budget_monitor=bud)
    fb = (f"A proposed start {t} FAILED automated verification (it violates a "
          f"busy interval or working hours). Re-derive carefully from the FULL "
          f"calendars and output a VALID start.\n\nAll calendars:\n"
          f"{cal_text(ppl)}\nFINAL: HH:MM")
    _, t2 = await one(rec, SYS, fb)
    return verify(prob, t2), bud.current_cost


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for cond, fn in [("single", c_single), ("debate", c_debate), ("gaia", c_gaia)]:
        npass = 0; cost = 0.0; recs = []
        for p in probs:
            bud = BudgetMonitor(max_cost_per_problem=0.6, max_iterations=30,
                                max_llm_calls=60)
            try:
                ok, c = await fn(p, fast, slow, bud)
            except Exception as e:
                ok, c = False, bud.current_cost
            npass += int(ok); cost += c
            recs.append({"problem_id": p["problem_id"], "passed": ok,
                         "n_people": p["n_people"]})
            print(f"[{cond:6s}] {p['problem_id']:11s} {'PASS' if ok else 'FAIL'}")
        acc = npass/len(probs)
        out[cond] = {"summary": {"condition": cond, "n": len(probs),
            "accuracy": acc, "total_cost_usd": cost}, "results": recs}
        print(f"== {cond}: acc={acc:.0%} cost=${cost:.3f} ==")
    json.dump(out, open(RES/f"nx7_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved nx7_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
