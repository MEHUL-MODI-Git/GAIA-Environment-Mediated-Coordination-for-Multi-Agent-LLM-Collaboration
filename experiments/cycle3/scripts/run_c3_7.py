#!/usr/bin/env python3
"""C3-7 — Emergence-steering causal test (does metacognition substitute for
conflict-as-task?).

C3-1 showed plain aggregation (vote/debate) is redundancy-dominated and scores
0% on synergy-required configs, while GAIA's conflict-as-task scores 100%.
2510.05174 reports that adding metacognitive "consider other agents" prompting
raises *measured cross-agent synergy*. Causal question: is prompt-level
metacognition a (cheaper) partial substitute for the structural
conflict-as-task mechanism — does it (a) raise measured synergy and (b)
recover synergy-required configs WITHOUT a reconciler?

13 E3 traps, 2 misled + 1 clean, NO reconciler (pure majority readout):
  A baseline      : standard misled/clean prompts, majority vote
  B metacognitive : every solver also gets a metacognitive preamble
                    ("other solvers may share a flawed heuristic; reason
                     independently; flag if your approach could be the common
                     trap") — applied to misled AND clean.
Outcome: majority accuracy + per-agent answers → fed to the C3-1 emergence
engine (synergy-required accuracy, config→outcome dependence). Honest either
way: if B↑ → metacognition is a partial prompt-level cure (cheap alt to
structure); if B≈A → conflict-as-task's structural edge is robust to
tactical prompting (mirrors MAST 'tactical fixes insufficient').
"""
import asyncio, json, os, sys, logging
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from groq import AsyncGroq  # fast/cheap open model; provider-independent
import re
DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle3"/"c3_7"/"results"
MODEL = "llama-3.3-70b-versatile"
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM

META = ("\n\nMETACOGNITIVE CHECK (do this first): other solvers may share a "
        "flawed heuristic or the same trap. Reason from first principles "
        "independently. If your method could be the *common mistake* for this "
        "kind of problem, say so explicitly and re-derive. Only then answer.")
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)


def parse(t):
    if not t:
        return None
    m = list(_I.finditer(t)) or re.findall(r"(-?\d[\d,]*)", t or "")
    try:
        return int((m[-1].group(1) if hasattr(m[-1], "group") else m[-1]).replace(",", "")) if m else None
    except Exception:
        return None


async def ask(c, sysm, um):
    for _ in range(4):
        try:
            r = await c.chat.completions.create(model=MODEL, temperature=0.0,
                max_tokens=900,
                messages=[{"role": "system", "content": sysm},
                          {"role": "user", "content": um}])
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                await asyncio.sleep(8); continue
            return ""
    return ""


async def episode(c, p, meta):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    mu = MisledSolverPrompts.format_user(q, h) + (META if meta else "")
    cu = MathSolverPrompts.format_user(q) + (META if meta else "")
    m0, m1, cl = await asyncio.gather(
        ask(c, MSYS, mu), ask(c, MSYS, mu), ask(c, CSYS, cu))
    a = [parse(m0), parse(m1), parse(cl)]
    v = [x for x in a if x is not None]
    fin = Counter(v).most_common(1)[0][0] if v else None
    return a, fin == t, t, p.get("common_wrong_answer")


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for cond, meta in [("baseline", False), ("metacognitive", True)]:
        recs = []; npass = 0
        for p in probs:
            try:
                a, ok, t, w = await asyncio.wait_for(episode(c, p, meta),
                                                     timeout=120)
            except Exception:
                a, ok, t, w = [None]*3, False, p["answer"], p.get("common_wrong_answer")
            npass += int(ok)
            recs.append({"problem_id": p["problem_id"],
                         "misled_answers": {"m0": a[0], "m1": a[1]},
                         "clean_answer": a[2], "passed": ok,
                         "ground_truth": t, "common_wrong_answer": w})
            print(f"[{cond:13s}] {p['problem_id']:22s} "
                  f"{'PASS' if ok else 'FAIL'} ans={a}")
        out[cond] = {"summary": {"condition": cond, "model": MODEL,
            "n": len(probs), "accuracy": npass/len(probs)}, "results": recs}
        print(f"== {cond}: acc={out[cond]['summary']['accuracy']:.0%} ==")
    json.dump(out, open(RES/f"c3_7_{ts}.json", "w"), indent=2, default=str)
    print(f"Saved c3_7_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
