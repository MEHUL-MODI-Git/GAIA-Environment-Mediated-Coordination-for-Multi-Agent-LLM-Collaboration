#!/usr/bin/env python3
"""NX3-open — cross-model generalization on an OPEN-WEIGHT model (Llama via Groq).

NX3 already showed the coordination gain holds across OpenAI families
(nano/4o-mini/3.5-turbo). The natural reviewer question: "does it hold on a
non-OpenAI, open-weight model?" Here the ENTIRE pool (solvers + reconciler)
is Llama-3.3-70B served by Groq — no OpenAI lineage at all. Raw Groq SDK is
used (mirrors the AnthropicLLM-wrapper lesson: bypass possibly-buggy wrappers
for a clean test).

Same 13 E3 traps, same 2-misled+1-clean correlated-failure stressor, same 3
conditions as NX1/E3:
  single        — 1 clean Llama solver (no stressor) → task solvable?
  majority_vote — 2 misled + 1 clean, majority → correlated failure bites?
  gaia          — + aggregator(conflict) + reconciler audit → recovered?
Independent of all OpenAI/Anthropic runs (separate provider, separate limit).
"""
import asyncio, json, os, re, sys, logging
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from groq import AsyncGroq
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
from gaia.prompts.math.reconciler import MathReconcilerPrompts

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = Path(__file__).parent.parent/"results"
MODEL = "llama-3.3-70b-versatile"          # full open-weight pool
CSYS = MathSolverPrompts.SYSTEM
MSYS = MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_INT = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)


def parse_ans(t):
    if not t:
        return None
    m = list(_INT.finditer(t))
    if m:
        try:
            return int(m[-1].group(1).replace(",", ""))
        except ValueError:
            return None
    m = re.findall(r"(-?\d[\d,]*)", t)
    try:
        return int(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


async def ask(client, sysm, um, t=0.0):
    for _ in range(4):                       # light retry on transient 429s
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=t, max_tokens=900,
                messages=[{"role": "system", "content": sysm},
                          {"role": "user", "content": um}])
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                await asyncio.sleep(8); continue
            return ""
    return ""


async def solve_clean(c, q):
    return await ask(c, CSYS, MathSolverPrompts.format_user(q))


async def solve_misled(c, q, h):
    return await ask(c, MSYS, MisledSolverPrompts.format_user(q, h))


async def cond_single(c, p):
    r = await solve_clean(c, p["question"])
    a = parse_ans(r)
    return a == p["answer"], a


async def cond_majority(c, p):
    q, h = p["question"], p["misleading_hint"]
    m0, m1, cl = await asyncio.gather(
        solve_misled(c, q, h), solve_misled(c, q, h), solve_clean(c, q))
    ans = [parse_ans(x) for x in (m0, m1, cl)]
    v = [x for x in ans if x is not None]
    fin = Counter(v).most_common(1)[0][0] if v else None
    return fin == p["answer"], fin


async def cond_gaia(c, p):
    q, h = p["question"], p["misleading_hint"]
    m0, m1, cl = await asyncio.gather(
        solve_misled(c, q, h), solve_misled(c, q, h), solve_clean(c, q))
    chains = [m0, m1, cl]; ans = [parse_ans(x) for x in chains]
    v = [x for x in ans if x is not None]
    if len(set(v)) <= 1:                     # no conflict
        return (v[0] == p["answer"] if v else False), (v[0] if v else None)
    nm = ["Solver-1", "Solver-2", "Solver-3"]
    summary = ", ".join(f"{nm[i]}={ans[i]}" for i in range(3))
    um = RP.format_user(q, summary, [(nm[i], chains[i]) for i in range(3)])
    rr = await ask(c, RP.SYSTEM, um)
    fin = parse_ans(rr)
    return fin == p["answer"], fin


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for cond, fn in [("single", cond_single), ("majority_vote", cond_majority),
                     ("gaia", cond_gaia)]:
        npass = 0; recs = []
        for p in probs:
            try:
                ok, fin = await asyncio.wait_for(fn(client, p), timeout=120)
            except Exception:
                ok, fin = False, None
            npass += int(ok)
            recs.append({"problem_id": p["problem_id"], "passed": ok,
                         "final": fin, "truth": p["answer"]})
            print(f"[{cond:13s}] {p['problem_id']:22s} "
                  f"{'PASS' if ok else 'FAIL'} fin={fin} truth={p['answer']}")
        out[cond] = {"summary": {"condition": cond, "model": MODEL,
            "n": len(probs), "accuracy": npass/len(probs)}, "results": recs}
        print(f"== {cond} ({MODEL}): acc={out[cond]['summary']['accuracy']:.0%} ==")
        json.dump(out, open(RES/f"nx3open_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved nx3open_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
