#!/usr/bin/env python3
"""C4-4 (OpenAI) — Leave-one-agent-out causal attribution (Shapley-ish).

Which role is load-bearing in GAIA's conflict-as-task pipeline? Re-run the
E3 GAIA-core on the 13 traps, ablating ONE agent at a time; accuracy drop =
that role's marginal contribution.

  full          : misled0 + misled1 + clean + reconciler
  -misled0      : misled1 + clean + reconciler   (still has a dissenter)
  -misled1      : misled0 + clean + reconciler
  -clean        : misled0 + misled1 + reconciler (NO dissenter — expect drop)
  -reconciler   : misled0 + misled1 + clean, majority (= C3-1 vote → ~0)

Hypothesis: clean-dissenter and reconciler are load-bearing; dropping either
collapses accuracy; dropping a redundant misled barely moves.

Ported off Groq → OpenAI gpt-4.1-nano (Groq hit an org-level 429 rate-limit
that produced empty-return artifacts — the same failure class that
contaminated C4-1 v1; not shipped). Same model as C4-1, so C4-4/C4-5 are
internally consistent with the compute-matched experiment. The leave-one-out
effect is a *model-internal relative* measure (Δacc within one pipeline), so
the provider switch is scientifically clean — footnoted honestly.
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
import openai
from gaia.prompts.math.solver import MathSolverPrompts
from gaia.prompts.math.misled_solver import MisledSolverPrompts
from gaia.prompts.math.reconciler import MathReconcilerPrompts

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle4"/"results"
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
_SEM = None  # created inside main() under the running loop (v4 lesson)


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


async def episode(c, p, drop):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    agents = []
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
        return (Counter(vals).most_common(1)[0][0] == t) if vals else False
    if len(set(vals)) <= 1:                          # no conflict → consensus
        return (vals[0] == t) if vals else False
    nm = list(chains)
    summ = ", ".join(f"{k}={ans[k]}" for k in nm)
    rr = await ask(c, RP.SYSTEM,
                   RP.format_user(q, summ, [(k, chains[k]) for k in nm]))
    return parse(rr) == t


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"model": MODEL}
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
        if isinstance(out[k], dict) and "accuracy" in out[k]:
            out[k]["marginal_drop_vs_full"] = round(base - out[k]["accuracy"], 3)
    json.dump(out, open(RES/f"c4_4_openai_{ts}.json", "w"), indent=2)
    print(f"Saved c4_4_openai_{ts}.json  (full={base:.0%})")
    for k, v in out.items():
        if k != "full" and isinstance(v, dict) and "marginal_drop_vs_full" in v:
            print(f"  drop {k}: Δacc = {v['marginal_drop_vs_full']:+.0%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
