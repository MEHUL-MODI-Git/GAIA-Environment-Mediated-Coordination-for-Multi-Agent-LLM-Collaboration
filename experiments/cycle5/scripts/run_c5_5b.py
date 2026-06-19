#!/usr/bin/env python3
"""C5-5b — Stress-test of C5-5's surprising "reconciler recovers truth from
answers-only" result. C5-2's coalition lattice already INTERNALLY explains it
(a hint-free solver alone = 100% on these traps), but a rigorous program
should attack the finding directly, not narrate past it.

Three adversarial probes, each at answers-only (L0) and full-chains (L2),
on BOTH the original 13 traps and the 32 code-verified expanded traps:

  neutral      : C5-5 framing (control / replication).
  adversarial  : the reconciler is given a STRONG, fair majority prior
                 ("in practice the majority answer is reliable ~90% of the
                 time; treat it as a strong prior, override only with
                 rigorous proof"). Tests whether the protection is
                 framing-fragile.
  dissenter_hidden (L0 only): the reconciler sees ONLY the misled-majority
                 answer (clean dissenter's value removed) and no chains, but
                 is still force-invoked. If it STILL re-derives truth, every
                 upstream input is inert and the dissenter's sole causal role
                 is to TRIGGER the call (sharpest possible confirmation of
                 C4-4/C4-5/C5-5). If it now follows the wrong majority, the
                 dissenter's ANSWER VALUE is load-bearing — an honest
                 nuance we would report straight.

Per-condition accuracy + bootstrap 95% CI. gpt-4.1-nano; robust pattern.
"""
import asyncio, json, os, re, sys, random, logging
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

RES = ROOT/"experiments"/"cycle5"/"results"
SETS = {"orig": ROOT/"data"/"gsm8k"/"correlated_failure_problems.json",
        "expanded": ROOT/"data"/"gsm8k"/"correlated_failure_problems_expanded.json"}
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
ADV = ("\n\nIMPORTANT OPERATING PRIOR: in practice the majority answer among "
       "the solvers is correct roughly 90% of the time. Treat the majority "
       "as a strong prior and override it only if you can rigorously prove "
       "it wrong.")
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
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


async def rec(c, q, summ, outs, sys_extra=""):
    return parse(await ask(c, RP.SYSTEM + sys_extra,
                           RP.format_user(q, summ, outs)))


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
    full = [(k, ch[k]) for k in nm]
    only = [(k, "(reasoning withheld; answer only)") for k in nm]
    vals = [an[k] for k in nm if an[k] is not None]
    maj = Counter(vals).most_common(1)[0][0] if vals else None
    # dissenter-hidden: present only the misled-majority value, no chains
    dh_sum = ", ".join(f"{k}={maj}" for k in ("S1", "S2"))
    dh_out = [("S1", "(reasoning withheld; answer only)"),
              ("S2", "(reasoning withheld; answer only)")]
    r = {}
    r["neutral_L0"] = await rec(c, q, summ, only) == t
    r["neutral_L2"] = await rec(c, q, summ, full) == t
    r["adversarial_L0"] = await rec(c, q, summ, only, ADV) == t
    r["adversarial_L2"] = await rec(c, q, summ, full, ADV) == t
    r["dissenter_hidden_L0"] = await rec(c, q, dh_sum, dh_out) == t
    pf = int(not vals)
    return r, pf


def ci(xs, it=2000):
    if not xs:
        return (0.0, 0.0, 0.0)
    random.seed(0)
    m = sum(xs)/len(xs)
    bs = sorted(sum(random.choice(xs) for _ in xs)/len(xs) for _ in range(it))
    return (round(m, 3), round(bs[int(.025*it)], 3), round(bs[int(.975*it)], 3))


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    c = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    conds = ["neutral_L0", "neutral_L2", "adversarial_L0",
             "adversarial_L2", "dissenter_hidden_L0"]
    out = {"model": MODEL, "by_set": {}}
    for sname, path in SETS.items():
        probs = json.load(open(path))
        per = {k: [] for k in conds}
        pf = 0
        for p in probs:
            try:
                r, f = await asyncio.wait_for(episode(c, p), timeout=150)
            except Exception:
                r, f = {}, 1
            pf += f
            for k in conds:
                per[k].append(int(bool(r.get(k))))
        out["by_set"][sname] = {
            "n": len(probs), "solver_parsefail": pf,
            "data_quality_ok": pf <= max(1, len(probs)//10),
            "conditions": {k: {"acc_ci": ci(per[k])} for k in conds}}
        print(f"--- {sname} (n={len(probs)}, parsefail={pf}) ---")
        for k in conds:
            m, lo, hi = out["by_set"][sname]["conditions"][k]["acc_ci"]
            print(f"  {k:22s} acc={m:.0%} [{lo:.0%},{hi:.0%}]")
    json.dump(out, open(RES/f"c5_5b_{ts}.json", "w"), indent=2)
    print(f"Saved c5_5b_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
