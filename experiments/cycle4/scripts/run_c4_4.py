#!/usr/bin/env python3
"""C4-4 — Leave-one-agent-out causal attribution (Shapley-ish marginal value).

Which role is load-bearing in GAIA's conflict-as-task pipeline? We re-run the
E3 GAIA-core on the 13 traps, ablating ONE agent at a time, and read off the
accuracy drop = that role's marginal contribution.

Pool (all Llama-3.3-70B via Groq, same model as NX3-open GAIA=100% reference):
  full          : misled0 + misled1 + clean + reconciler
  -misled0      : misled1 + clean + reconciler   (still has a dissenter)
  -misled1      : misled0 + clean + reconciler
  -clean        : misled0 + misled1 + reconciler (NO dissenter — expect drop)
  -reconciler   : misled0 + misled1 + clean, majority (= the C3-1 vote → ~0)
Hypothesis: clean-dissenter and reconciler are the load-bearing roles;
dropping either collapses accuracy; dropping a redundant misled barely moves.
Bounded (fixed calls, retry-capped) — no hang risk.
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
RES = ROOT/"experiments"/"cycle4"/"results"
MODEL = "llama-3.3-70b-versatile"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)


def parse(t):
    if not t:
        return None
    m = list(_I.finditer(t)) or re.findall(r"(-?\d[\d,]*)", t or "")
    try:
        return int((m[-1].group(1) if hasattr(m[-1], "group") else m[-1]).replace(",", "")) if m else None
    except Exception:
        return None


async def ask(c, s, u, t=0.0):
    for _ in range(4):
        try:
            r = await c.chat.completions.create(model=MODEL, temperature=t,
                max_tokens=900, messages=[{"role": "system", "content": s},
                                          {"role": "user", "content": u}])
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                await asyncio.sleep(8); continue
            return ""
    return ""


async def episode(c, p, drop):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    agents = []   # (name, sys, user, kind)
    for nm in ("misled0", "misled1"):
        if nm != drop:
            agents.append((nm, MSYS, MisledSolverPrompts.format_user(q, h)))
    if drop != "clean":
        agents.append(("clean", CSYS, MathSolverPrompts.format_user(q)))
    outs = await asyncio.gather(*[ask(c, s, u) for _, s, u in agents])
    chains = {agents[i][0]: outs[i] for i in range(len(agents))}
    ans = {k: parse(v) for k, v in chains.items()}
    vals = [a for a in ans.values() if a is not None]
    if drop == "reconciler":                       # majority readout
        fin = Counter(vals).most_common(1)[0][0] if vals else None
        return fin == t
    if len(set(vals)) <= 1:                          # no conflict → consensus
        return (vals[0] == t) if vals else False
    nm = list(chains)
    summ = ", ".join(f"{k}={ans[k]}" for k in nm)
    rr = await ask(c, RP.SYSTEM,
                   RP.format_user(q, summ, [(k, chains[k]) for k in nm]))
    return parse(rr) == t


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    for drop in (None, "misled0", "misled1", "clean", "reconciler"):
        key = "full" if drop is None else f"-{drop}"
        npass = 0
        for p in probs:
            try:
                ok = await asyncio.wait_for(episode(c, p, drop), timeout=120)
            except Exception:
                ok = False
            npass += int(ok)
        acc = npass/len(probs)
        out[key] = {"drop": drop, "n": len(probs), "accuracy": round(acc, 3)}
        print(f"[{key:12s}] acc={acc:.0%}")
    base = out["full"]["accuracy"]
    for k in out:
        out[k]["marginal_drop_vs_full"] = round(base - out[k]["accuracy"], 3)
    json.dump(out, open(RES/f"c4_4_{ts}.json", "w"), indent=2)
    print(f"Saved c4_4_{ts}.json  (full={base:.0%})")
    for k, v in out.items():
        if k != "full":
            print(f"  drop {k}: Δacc = {v['marginal_drop_vs_full']:+.0%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
