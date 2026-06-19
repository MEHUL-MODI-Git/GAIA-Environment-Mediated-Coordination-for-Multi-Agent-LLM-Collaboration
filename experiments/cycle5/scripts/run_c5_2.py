#!/usr/bin/env python3
"""C5-2 — Exact Shapley attribution of agent roles (rigorous upgrade of C4-4).

C4-4 used leave-one-out: a single-permutation marginal proxy. The 2026
standard (SHARP 2602.08335, AgentSHAP 2512.12597, Shapley-Coop 2506.07388) is
the game-theoretic Shapley value: marginal contribution averaged over ALL
coalition orderings. We compute it EXACTLY (n=4 ⇒ 16 coalitions, tractable).

Game: players N = {misled0, misled1, clean, reconciler}. Characteristic
function v(S) = accuracy over the 13 E3 traps of the pipeline restricted to
coalition S, with C4-4-consistent semantics:
  - solver chains generated ONCE per trap and reused across all coalitions
    (Shapley over fixed realizations — removes sampling noise, the clean
    estimator);
  - reconciler ∈ S and the present solvers conflict (≥2 distinct answers)
    ⇒ reconcile; reconciler ∈ S but no conflict ⇒ consensus;
  - reconciler ∉ S ⇒ majority vote over present solvers;
  - no present solver ⇒ no answer (v contribution 0); v(∅)=0.

Outputs: exact Shapley φ_i per role; the efficiency-axiom check
Σφ_i = v(N)−v(∅); the pairwise interaction index I(clean,reconciler).
Honest: φ is defined on THIS game (binary trap accuracy, fixed chains);
reported as role credit, not a universal constant.

gpt-4.1-nano (same as C4-x). Bounded: 13×3 solver calls + ≤ memoized
reconcile calls. Robust pattern; data-quality guard.
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
                r = await c.chat.completions.create(model=MODEL,
                    temperature=t, max_tokens=900,
                    messages=[{"role": "system", "content": s},
                              {"role": "user", "content": u}])
                txt = r.choices[0].message.content
                if txt and txt.strip():
                    return txt
                await asyncio.sleep(2)
            except Exception as e:
                await asyncio.sleep(5 if "rate" in str(e).lower() else 2)
        return ""


async def gen_chains(c, probs):
    """One solver realization per trap, reused across all coalitions."""
    rows = []
    for p in probs:
        q, h, t = p["question"], p["misleading_hint"], p["answer"]
        m0, m1, cl = await asyncio.gather(
            ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
            ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
            ask(c, CSYS, MathSolverPrompts.format_user(q)))
        ch = {"misled0": m0, "misled1": m1, "clean": cl}
        rows.append({"q": q, "t": t,
                     "ch": ch, "an": {k: parse(v) for k, v in ch.items()}})
    return rows


async def pred_for(c, row, S, rcache):
    """Restricted-pipeline prediction for coalition S on one trap."""
    present = [r for r in SOLVERS if r in S]
    if not present:
        return None                              # nothing to answer with
    an = row["an"]; vals = [an[r] for r in present if an[r] is not None]
    if "reconciler" in S:
        if len(set(vals)) >= 2:                   # genuine conflict → reconcile
            key = (id(row), frozenset(present))
            if key not in rcache:
                summ = ", ".join(f"{r}={an[r]}" for r in present)
                outs = [(r, row["ch"][r]) for r in present]
                rcache[key] = parse(await ask(
                    c, RP.SYSTEM, RP.format_user(row["q"], summ, outs)))
            return rcache[key]
        return vals[0] if vals else None          # consensus
    return Counter(vals).most_common(1)[0][0] if vals else None  # majority


async def v_of(c, S, rows, rcache):
    if not S:
        return 0.0
    ok = 0
    for row in rows:
        pred = await pred_for(c, row, set(S), rcache)
        ok += int(pred == row["t"])
    return ok/len(rows)


def shapley(vals, players):
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}
    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for k in range(len(others)+1):
            for S in combinations(others, k):
                w = factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
                phi[p] += w*(vals[frozenset(S+(p,))] - vals[frozenset(S)])
    return phi


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = await gen_chains(c, probs)
    pf = sum(1 for r in rows if not [a for a in r["an"].values()
                                     if a is not None])
    rcache = {}
    vals = {}
    for k in range(len(PLAYERS)+1):
        for S in combinations(PLAYERS, k):
            vals[frozenset(S)] = await v_of(c, S, rows, rcache)
    phi = shapley(vals, PLAYERS)
    grand = vals[frozenset(PLAYERS)]; empty = vals[frozenset()]
    eff_lhs = round(sum(phi.values()), 6)
    eff_rhs = round(grand - empty, 6)
    # pairwise interaction I(clean, reconciler)
    i, j = "clean", "reconciler"
    rest = [p for p in PLAYERS if p not in (i, j)]
    n = len(PLAYERS); inter = 0.0
    for k in range(len(rest)+1):
        for S in combinations(rest, k):
            w = factorial(len(S))*factorial(n-len(S)-2)/factorial(n-1)
            fs = frozenset(S)
            inter += w*(vals[fs | {i, j}] - vals[fs | {i}]
                        - vals[fs | {j}] + vals[fs])
    out = {"model": MODEL, "n_traps": len(rows), "solver_parsefail": pf,
           "data_quality_ok": pf <= 1,
           "shapley": {p: round(phi[p], 4) for p in PLAYERS},
           "efficiency_axiom": {"sum_phi": eff_lhs,
                                "v_grand_minus_empty": eff_rhs,
                                "holds": abs(eff_lhs-eff_rhs) < 1e-6},
           "v_grand": grand, "v_empty": empty,
           "interaction_clean_reconciler": round(inter, 4),
           "coalitions": {",".join(sorted(S)) or "(empty)": round(vals[frozenset(S)], 3)
                          for k in range(len(PLAYERS)+1)
                          for S in combinations(PLAYERS, k)}}
    json.dump(out, open(RES/f"c5_2_{ts}.json", "w"), indent=2)
    print("Exact Shapley φ (role credit on the 13-trap game):")
    for p in PLAYERS:
        print(f"  {p:12s} φ = {phi[p]:+.4f}")
    print(f"efficiency: Σφ={eff_lhs} vs v(N)-v(∅)={eff_rhs} "
          f"holds={out['efficiency_axiom']['holds']}")
    print(f"interaction I(clean,reconciler) = {inter:+.4f}")
    print(f"dq_ok={out['data_quality_ok']} (parsefail={pf})")
    print(f"Saved c5_2_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
