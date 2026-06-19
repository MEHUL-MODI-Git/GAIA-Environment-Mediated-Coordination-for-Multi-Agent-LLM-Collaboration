#!/usr/bin/env python3
"""C5-xmodel — Cross-model replication on a DIFFERENT family (Anthropic
claude-haiku-4-5, raw SDK — the gaia anthropic wrapper is buggy) of the two
CYCLE-5 results whose generalization most matters for "is the mechanism
model-specific?":

  (A) C5-2 exact Shapley game (16 coalitions): does clean+reconciler still
      carry the credit with strong positive interaction on a different model?
  (B) C5-5 reconciler info-bottleneck (L0 answers-only vs L2 full-chains):
      does answers-only still saturate, i.e. is the reconciler a hint-free
      re-solver on this model too?

Canonical 13 traps (direct comparability with the gpt-4.1-nano numbers).
Same coalition/level semantics as C5-2 / C5-5. Robust pattern
(semaphore-inside-main, empty-retry, episode guards, data-quality guard).
Honest: if the mechanism does NOT transfer (e.g. answers-only collapses on
haiku, or interaction is weak), that is reported straight — a cross-model
boundary is a finding, not a failure.
"""
import asyncio, json, os, re, sys, logging
from collections import Counter
from itertools import combinations
from math import factorial
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import anthropic
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
from gaia.prompts.math.reconciler import MathReconcilerPrompts

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle5"/"results"
MODEL = "claude-haiku-4-5"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
PLAYERS = ("misled0", "misled1", "clean", "reconciler")
SOLVERS = ("misled0", "misled1", "clean")
_SEM = None


def parse(t):
    if not t:
        return None
    m = list(_I.finditer(t)) or re.findall(r"(-?\d[\d,]*)", t or "")
    try:
        return int((m[-1].group(1) if hasattr(m[-1], "group") else m[-1]).replace(",", "")) if m else None
    except Exception:
        return None


async def ask(c, s, u, t=0.0):
    async with _SEM:
        for _ in range(5):
            try:
                r = await c.messages.create(model=MODEL, max_tokens=900,
                    temperature=t, system=s,
                    messages=[{"role": "user", "content": u}])
                txt = "".join(b.text for b in r.content
                              if getattr(b, "type", "") == "text")
                if txt and txt.strip():
                    return txt
                await asyncio.sleep(2)
            except Exception as e:
                await asyncio.sleep(6 if "rate" in str(e).lower()
                                    or "overload" in str(e).lower() else 2)
        return ""


async def gen(c, probs):
    rows = []
    for p in probs:
        q, h, t = p["question"], p["misleading_hint"], p["answer"]
        m0, m1, cl = await asyncio.gather(
            ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
            ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
            ask(c, CSYS, MathSolverPrompts.format_user(q)))
        ch = {"misled0": m0, "misled1": m1, "clean": cl}
        rows.append({"q": q, "t": t, "ch": ch,
                     "an": {k: parse(v) for k, v in ch.items()}})
    return rows


async def pred(c, row, S, cache):
    present = [r for r in SOLVERS if r in S]
    if not present:
        return None
    an = row["an"]; vals = [an[r] for r in present if an[r] is not None]
    if "reconciler" in S:
        if len(set(vals)) >= 2:
            key = (id(row), frozenset(present))
            if key not in cache:
                summ = ", ".join(f"{r}={an[r]}" for r in present)
                cache[key] = parse(await ask(c, RP.SYSTEM, RP.format_user(
                    row["q"], summ, [(r, row["ch"][r]) for r in present])))
            return cache[key]
        return vals[0] if vals else None
    return Counter(vals).most_common(1)[0][0] if vals else None


async def v_of(c, S, rows, cache):
    if not S:
        return 0.0
    ok = 0
    for r in rows:
        ok += int(await pred(c, r, set(S), cache) == r["t"])
    return ok/len(rows)


def shapley(vals, players):
    n = len(players); phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for k in range(len(others)+1):
            for S in combinations(others, k):
                w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                phi[p] += w*(vals[frozenset(S+(p,))]-vals[frozenset(S)])
    return phi


async def reconcile_level(c, row, withhold):
    nm = ["S1", "S2", "S3"]
    mp = {"S1": "misled0", "S2": "misled1", "S3": "clean"}
    an = {k: row["an"][mp[k]] for k in nm}
    summ = ", ".join(f"{k}={an[k]}" for k in nm)
    outs = [(k, "(reasoning withheld; answer only)" if withhold
             else row["ch"][mp[k]]) for k in nm]
    return parse(await ask(c, RP.SYSTEM, RP.format_user(row["q"], summ, outs)))


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(4)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = await gen(c, probs)
    pf = sum(1 for r in rows if not [a for a in r["an"].values()
                                     if a is not None])
    cache = {}
    vals = {}
    for k in range(len(PLAYERS)+1):
        for S in combinations(PLAYERS, k):
            vals[frozenset(S)] = await v_of(c, S, rows, cache)
    phi = shapley(vals, PLAYERS)
    grand, empty = vals[frozenset(PLAYERS)], vals[frozenset()]
    i, j = "clean", "reconciler"
    rest = [p for p in PLAYERS if p not in (i, j)]; n = len(PLAYERS)
    inter = 0.0
    for k in range(len(rest)+1):
        for S in combinations(rest, k):
            w = factorial(len(S))*factorial(n-len(S)-2)/factorial(n-1)
            fs = frozenset(S)
            inter += w*(vals[fs | {i, j}]-vals[fs | {i}]
                        - vals[fs | {j}]+vals[fs])
    # info-bottleneck L0 vs L2
    o0 = o2 = 0
    for r in rows:
        o0 += int(await reconcile_level(c, r, True) == r["t"])
        o2 += int(await reconcile_level(c, r, False) == r["t"])
    l0 = o0/len(rows); l2 = o2/len(rows)
    out = {"model": MODEL, "n_traps": len(rows), "solver_parsefail": pf,
           "data_quality_ok": pf <= 1,
           "shapley": {p: round(phi[p], 4) for p in PLAYERS},
           "efficiency_holds": abs(sum(phi.values())-(grand-empty)) < 1e-6,
           "v_grand": grand, "v_empty": empty,
           "interaction_clean_reconciler": round(inter, 4),
           "info_bottleneck": {"L0_answers_only": round(l0, 3),
                               "L2_full_chains": round(l2, 3)},
           "coalitions": {",".join(sorted(S)) or "(empty)": round(vals[frozenset(S)], 3)
                          for k in range(len(PLAYERS)+1)
                          for S in combinations(PLAYERS, k)}}
    json.dump(out, open(RES/f"c5_xmodel_haiku_{ts}.json", "w"), indent=2)
    print(f"[{MODEL}] Shapley:",
          {p: round(phi[p], 3) for p in PLAYERS})
    print(f"  interaction(clean,reconciler)={inter:+.4f} "
          f"efficiency_holds={out['efficiency_holds']}")
    print(f"  info-bottleneck L0_answers_only={l0:.0%} "
          f"L2_full_chains={l2:.0%}")
    print(f"  dq_ok={out['data_quality_ok']} (parsefail={pf})")
    print(f"Saved c5_xmodel_haiku_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
