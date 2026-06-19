"""C4-5 — Counterfactual transcript intervention (a do-operator on coordination).

A genuinely new KIND of experiment: instead of removing an agent (C4-4) we
SURGICALLY EDIT one agent's posted reasoning in an otherwise-real episode and
replay ONLY the reconciler — isolating its *causal sensitivity* to each input
(Pearl's do-operator at the transcript level). On the 13 E3 traps,
Llama-3.3-70B:

  baseline   : 2 misled + 1 clean → reconciler                (expect ~100%)
  do(clean:=trap)   : overwrite the CLEAN agent's chain with a misled/trap
                      chain → now ALL THREE say the wrong answer. Does the
                      reconciler still recover truth, or does it follow a
                      unanimous-but-wrong board?  (tests reliance on the
                      dissenter)
  do(misled0:=correct): overwrite ONE misled chain with the clean/correct
                      chain → 2 correct + 1 misled. Does the reconciler track
                      the new majority? (tests it isn't just contrarian)
  do(clean:=empty)  : blank the clean chain (present but uninformative).
                      Tests whether the dissent VALUE or merely its PRESENCE
                      matters.

This decomposes *what the reconciler is actually causally using* — a
mechanistic probe no prior experiment performs (leave-one-out removes;
this perturbs content while holding structure fixed). Bounded, Groq.
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


async def reconcile(c, q, chains, ans):
    nm = list(chains)
    summ = ", ".join(f"{k}={ans[k]}" for k in nm)
    return parse(await ask(c, RP.SYSTEM,
                 RP.format_user(q, summ, [(k, chains[k]) for k in nm])))


async def episode(c, p):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    m0, m1, cl = await asyncio.gather(
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, CSYS, MathSolverPrompts.format_user(q)))
    base_ch = {"S1": m0, "S2": m1, "S3": cl}
    base_an = {k: parse(v) for k, v in base_ch.items()}
    out = {}
    # baseline
    out["baseline"] = await reconcile(c, q, base_ch, base_an) == t
    # do(clean := trap): clean chain replaced by a misled chain → all-wrong board
    ch = dict(base_ch); ch["S3"] = m0
    an = {k: parse(v) for k, v in ch.items()}
    out["do_clean=trap"] = await reconcile(c, q, ch, an) == t
    # do(misled0 := correct): one misled replaced by the clean chain
    ch = dict(base_ch); ch["S1"] = cl
    an = {k: parse(v) for k, v in ch.items()}
    out["do_misled0=correct"] = await reconcile(c, q, ch, an) == t
    # do(clean := empty): dissent present but uninformative
    ch = dict(base_ch); ch["S3"] = "(no reasoning provided)"
    an = {"S1": base_an["S1"], "S2": base_an["S2"], "S3": None}
    out["do_clean=empty"] = await reconcile(c, q, ch, an) == t
    return out


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg = {}
    n = len(probs)
    for p in probs:
        try:
            r = await asyncio.wait_for(episode(c, p), timeout=150)
        except Exception:
            r = {}
        for k, v in r.items():
            agg.setdefault(k, 0)
            agg[k] += int(bool(v))
    out = {k: {"accuracy": round(agg[k]/n, 3), "n": n} for k in agg}
    json.dump(out, open(RES/f"c4_5_{ts}.json", "w"), indent=2)
    for k, v in out.items():
        print(f"[{k:22s}] acc={v['accuracy']:.0%}")
    print(f"Saved c4_5_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
