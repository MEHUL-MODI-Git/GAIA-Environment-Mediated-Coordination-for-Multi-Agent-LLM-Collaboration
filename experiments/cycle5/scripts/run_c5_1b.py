#!/usr/bin/env python3
"""C5-1b — The decisive experiment: WHEN does multi-agent independence beat
single-agent self-refinement? (Isolates GAIA's distinct value after C5-1
showed unbiased self-refine ties GAIA at equal tokens.)

C5-1 honest finding: at matched reasoning tokens, single-agent self-refine
(draft→critique→revise) = GAIA = 100%, while self-consistency / extended =
23%. So the active ingredient is STRUCTURED self-correction, not multi-agent
multiplicity — on a substrate where the lone agent's self-critique is
UNBIASED. Real correlated-failure deployments are exactly the case where the
self-critique SHARES the bias (the same misleading procedure contaminates
both the draft and the self-review). C5-1b tests that, token-matched, 13
traps (+ 32 verified expanded):

  gaia_core          : 2 misled + 1 clean → INDEPENDENT hint-free reconciler.
  sa_refine_unbiased : draft→critique→revise, critic does NOT see the hint
                       (= C5-1's sa_refine; expected ≈100%).
  sa_refine_biased   : draft→critique→revise, critic ALSO carries the
                       misleading hint as an authoritative procedure
                       (self-critique correlated with the original error).
  debate_biased      : 3 agents ALL see the hint, 2 rounds, majority
                       (fully correlated panel — control).

Hypothesis: sa_refine_biased COLLAPSES (a single agent cannot escape a bias
its own self-critique shares) while gaia_core HOLDS (its reconciler is
structurally INDEPENDENT of the bias). That gap is GAIA's distinct,
honestly-bounded value: independence of the corrector from the error.
Honest: if biased self-refine does NOT collapse, GAIA's accuracy advantage
on this substrate is genuinely marginal and we say so.

gpt-4.1-nano; robust pattern; bootstrap CIs.
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

SETS = {"orig": ROOT/"data"/"gsm8k"/"correlated_failure_problems.json",
        "expanded": ROOT/"data"/"gsm8k"/"correlated_failure_problems_expanded.json"}
RES = ROOT/"experiments"/"cycle5"/"results"
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
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


async def ask(c, s, u, t=0.0, mx=900):
    async with _SEM:
        for _ in range(5):
            try:
                r = await c.chat.completions.create(model=MODEL,
                    temperature=t, max_tokens=mx,
                    messages=[{"role": "system", "content": s},
                              {"role": "user", "content": u}])
                txt = r.choices[0].message.content
                tok = r.usage.completion_tokens if r.usage else 0
                if txt and txt.strip():
                    return txt, tok
                await asyncio.sleep(2)
            except Exception as e:
                await asyncio.sleep(5 if "rate" in str(e).lower() else 2)
        return "", 0


async def gaia_core(c, p):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    (m0, k0), (m1, k1), (cl, kc) = await asyncio.gather(
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h)),
        ask(c, CSYS, MathSolverPrompts.format_user(q)))
    tok = k0+k1+kc
    an = {"S1": parse(m0), "S2": parse(m1), "S3": parse(cl)}
    vals = [v for v in an.values() if v is not None]
    if len(set(vals)) >= 2:
        summ = ", ".join(f"{k}={an[k]}" for k in an)
        rr, kr = await ask(c, RP.SYSTEM, RP.format_user(
            q, summ, [("S1", m0), ("S2", m1), ("S3", cl)]))
        tok += kr; pred = parse(rr)
    else:
        pred = vals[0] if vals else None
    return int(pred == t), tok


async def refine(c, p, B, biased):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    cur, tok = await ask(c, MSYS, MisledSolverPrompts.format_user(q, h))
    it = 0
    while tok < B and it < 6:
        if biased:
            cu = (f"Problem: {q}\n\nRecall the established procedure you must "
                  f"follow: {h}\n\nYour current solution:\n{cur}\n\n"
                  "Re-examine your solution for consistency WITH THAT "
                  "PROCEDURE and correctness; produce a corrected full "
                  "solution ending **Final Answer: N** (or restate if "
                  "already correct).")
        else:
            cu = (f"Problem: {q}\n\nYour current solution:\n{cur}\n\n"
                  "Critically re-examine it from first principles. Are any "
                  "assumptions, formulas, endpoints, or arithmetic steps "
                  "wrong? Produce a corrected full solution ending "
                  "**Final Answer: N** (or restate if already correct).")
        cur, k = await ask(c, MSYS, cu)
        tok += k; it += 1
        if not cur:
            break
    return int(parse(cur) == t), tok


async def debate_biased(c, p, B, rounds=2):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    base = (f"Problem: {q}\n\nEstablished procedure to apply: {h}\n"
            "End with **Final Answer: N**.")
    last = [""]*3; tok = 0
    for r in range(rounds):
        peer = ("" if r == 0 else "\n\nPeers said: " +
                "; ".join(f"A{i+1}={parse(last[i])}" for i in range(3)))
        outs = await asyncio.gather(*[
            ask(c, MSYS, base + peer) for _ in range(3)])
        for i, (txt, k) in enumerate(outs):
            last[i] = txt; tok += k
    vals = [parse(x) for x in last if parse(x) is not None]
    pred = Counter(vals).most_common(1)[0][0] if vals else None
    return int(pred == t), tok


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
    out = {"model": MODEL, "by_set": {}}
    for sname, path in SETS.items():
        probs = json.load(open(path))
        n = len(probs)
        g_ok = []; g_tok = []
        for p in probs:
            try:
                ok, tk = await asyncio.wait_for(gaia_core(c, p), timeout=150)
            except Exception:
                ok, tk = 0, 0
            g_ok.append(ok); g_tok.append(tk)
        valid = [x for x in g_tok if x > 0]
        B = sum(valid)/len(valid) if valid else 1200
        arms = {
            "gaia_core": [g_ok, g_tok],
            "sa_refine_unbiased": None,
            "sa_refine_biased": None,
            "debate_biased": None}
        for name in ("sa_refine_unbiased", "sa_refine_biased",
                     "debate_biased"):
            oks = []; tks = []
            for p in probs:
                try:
                    if name == "debate_biased":
                        ok, tk = await asyncio.wait_for(
                            debate_biased(c, p, B), timeout=200)
                    else:
                        ok, tk = await asyncio.wait_for(
                            refine(c, p, B,
                                   biased=(name == "sa_refine_biased")),
                            timeout=200)
                except Exception:
                    ok, tk = 0, 0
                oks.append(ok); tks.append(tk)
            arms[name] = [oks, tks]
        rec = {}
        for name, (oks, tks) in arms.items():
            sv = [x for x in tks if x > 0]
            rec[name] = {"acc_ci": ci(oks),
                         "mean_tokens": round(sum(sv)/len(sv), 1) if sv else 0}
        out["by_set"][sname] = {"n": n, "budget_B": round(B, 1),
                                "data_quality_ok": len(valid) >= n-1,
                                "arms": rec}
        print(f"--- {sname} (n={n}, B≈{B:.0f}) ---")
        for name in arms:
            a = rec[name]
            print(f"  {name:20s} acc={a['acc_ci'][0]:.0%} "
                  f"[{a['acc_ci'][1]:.0%},{a['acc_ci'][2]:.0%}] "
                  f"tok≈{a['mean_tokens']:.0f}")
    json.dump(out, open(RES/f"c5_1b_{ts}.json", "w"), indent=2)
    print(f"Saved c5_1b_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
