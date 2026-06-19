#!/usr/bin/env python3
"""C5-5 — Reconciler information-bottleneck (closes deferred W6 rigorously).

What is the MINIMAL sufficient input the reconciler needs to recover truth on
a 2-misled/1-clean correlated-failure board? We generate the 3 solver chains
ONCE per trap, then replay the reconciler with progressively richer input and
read accuracy vs information level — an information-bottleneck on the
reconciler's input.

  L0 answers_only : conflict summary only; chains withheld.
  L1 +final_lines : + each solver's one-line final-answer statement.
  L2 +full_chains : + each solver's complete reasoning chain (standard GAIA).
  L3 +conflict_sig: L2 + an explicit typed CONFLICT signal annotation
                    (majority value vs dissenter) — GAIA's structured signal.

The smallest L at which accuracy saturates = the reconciler's minimal
sufficient statistic. Hypothesis (from C4-5: reconciler is an independent
re-deriver, not a vote-follower): answers alone are NOT sufficient; the
reasoning content is the load-bearing input; the explicit signal adds little
on top of full chains. Honest: if L0 already saturates, the reconciler is
just re-solving cold and chain content is inert — we report that straight.

Same model as C4-x (gpt-4.1-nano) ⇒ internally consistent. Bounded
(7 calls/problem), OpenAI; robust pattern (semaphore-inside-main,
empty-retry, episode timeout, data-quality guard).
"""
import asyncio, json, os, re, sys, logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import openai
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
from gaia.prompts.math.reconciler import MathReconcilerPrompts

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle5"/"results"
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
LEVELS = ["L0_answers_only", "L1_final_lines", "L2_full_chains",
          "L3_conflict_sig"]
_SEM = None


def parse(t):
    if not t:
        return None
    m = list(_I.finditer(t)) or re.findall(r"(-?\d[\d,]*)", t or "")
    try:
        return int((m[-1].group(1) if hasattr(m[-1], "group") else m[-1]).replace(",", "")) if m else None
    except Exception:
        return None


def final_line(t):
    if not t:
        return "(no answer)"
    m = list(_I.finditer(t))
    if m:
        return m[-1].group(0)
    a = parse(t)
    return f"**Final Answer: {a}**" if a is not None else "(no answer)"


async def ask(c, s, u, t=0.0, mx=900):
    async with _SEM:
        for _ in range(5):
            try:
                r = await c.chat.completions.create(model=MODEL,
                    temperature=t, max_tokens=mx,
                    messages=[{"role": "system", "content": s},
                              {"role": "user", "content": u}])
                txt = r.choices[0].message.content
                if txt and txt.strip():
                    return txt
                await asyncio.sleep(2)
            except Exception as e:
                await asyncio.sleep(5 if "rate" in str(e).lower() else 2)
        return ""


async def reconcile(c, q, summ, outs):
    return parse(await ask(c, RP.SYSTEM, RP.format_user(q, summ, outs)))


async def episode(c, p):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    m0, m1, cl = await asyncio.gather(
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, CSYS, MathSolverPrompts.format_user(q)))
    ch = {"S1": m0, "S2": m1, "S3": cl}
    an = {k: parse(v) for k, v in ch.items()}
    nm = list(ch)
    summ = ", ".join(f"{k}={an[k]}" for k in nm)
    # majority vs dissenter for the explicit signal annotation
    from collections import Counter
    vals = [an[k] for k in nm if an[k] is not None]
    maj = Counter(vals).most_common(1)[0][0] if vals else None
    sig = (f"[CONFLICT signal | severity=0.9 | majority_answer={maj} | "
           f"dissenter(s)=" +
           ",".join(f"{k}({an[k]})" for k in nm if an[k] != maj) + "] ")
    res = {}
    res["L0_answers_only"] = await reconcile(
        c, q, summ, [(k, "(reasoning withheld; answer only)") for k in nm]) == t
    res["L1_final_lines"] = await reconcile(
        c, q, summ, [(k, final_line(ch[k])) for k in nm]) == t
    res["L2_full_chains"] = await reconcile(
        c, q, summ, [(k, ch[k]) for k in nm]) == t
    res["L3_conflict_sig"] = await reconcile(
        c, q, sig + summ, [(k, ch[k]) for k in nm]) == t
    parsefail = int(not vals)               # all-solver-unparseable guard
    return res, parsefail


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg = {L: 0 for L in LEVELS}
    pf = 0
    n = len(probs)
    for p in probs:
        try:
            r, fail = await asyncio.wait_for(episode(c, p), timeout=150)
        except Exception:
            r, fail = {}, 1
        pf += fail
        for L in LEVELS:
            agg[L] += int(bool(r.get(L)))
    out = {"model": MODEL, "n": n, "solver_parsefail": pf,
           "data_quality_ok": pf <= 1,
           "by_level": {L: {"accuracy": round(agg[L]/n, 3),
                            "info_level": i}
                        for i, L in enumerate(LEVELS)}}
    # minimal sufficient = first level reaching ≥0.95 of the max
    mx = max(v["accuracy"] for v in out["by_level"].values())
    out["max_acc"] = mx
    out["minimal_sufficient_level"] = next(
        (L for L in LEVELS if out["by_level"][L]["accuracy"] >= 0.95*mx and mx > 0),
        None)
    json.dump(out, open(RES/f"c5_5_{ts}.json", "w"), indent=2)
    for L in LEVELS:
        print(f"[{L:18s}] acc={out['by_level'][L]['accuracy']:.0%}")
    print(f"minimal_sufficient={out['minimal_sufficient_level']} "
          f"dq_ok={out['data_quality_ok']} (parsefail={pf})")
    print(f"Saved c5_5_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
