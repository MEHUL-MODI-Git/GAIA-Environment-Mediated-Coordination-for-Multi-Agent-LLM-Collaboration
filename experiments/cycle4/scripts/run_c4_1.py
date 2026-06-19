#!/usr/bin/env python3
"""C4-1 — Compute-matched baseline (the paper-defining rebuttal experiment).

Reviewer objection (Budget-Aware Eval, EMNLP'24): compute-matched
self-consistency beats multi-agent systems → "is GAIA just more compute?"
Answer, under GAIA's TARGET regime (correlated misleading information):

  Resampling a biased reasoner reproduces the bias — self-consistency
  accuracy is FLAT in compute (C3-1: traps are synergy-required, vote reads
  only redundancy). GAIA's STRUCTURAL conflict-as-task breaks the
  correlation. So extra compute provably cannot substitute for structure
  here.

All-Llama-3.3-70B (Groq) so every method uses the SAME base model and we can
plot accuracy vs total LLM-calls honestly. 13 E3 traps.

Arms (N-swept budget):
  sc_misled@N  : N indep samples of a MISLED solver (temp 0.7), majority.
                 Expect ≈0% and FLAT in N (the key curve).
  sc_clean@N   : N indep samples of a CLEAN solver, majority.
                 Expect ≈100% — proves traps are individually solvable, so
                 the failure is correlated BIAS not trap hardness (control).
  bestof_misled@N: N misled samples, pick the one with highest self-rated
                 confidence. Expect still ≈0% (misled = confidently wrong;
                 ties to C4-3 calibration).
  GAIA-llama   : reuse NX3-open GAIA-Llama = 100% at its real ~6-call budget
                 (same model) — the matched structural point.

N ∈ {1,3,6,9}. GAIA's mean budget ≈ 6 calls → compare at N=6. The money
figure: x=total LLM-calls, y=accuracy; sc_misled flat ≈0 across N; GAIA a
single point at (≈6, 100%).
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

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle4"/"results"
MODEL = "llama-3.3-70b-versatile"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
CONF = ("\n\nAfter your answer add a line exactly: CONFIDENCE: X "
        "where X is your 0-1 confidence.")
_C = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.I)


def parse(t):
    if not t:
        return None
    m = list(_I.finditer(t)) or re.findall(r"(-?\d[\d,]*)", t or "")
    try:
        return int((m[-1].group(1) if hasattr(m[-1], "group") else m[-1]).replace(",", "")) if m else None
    except Exception:
        return None


def conf(t):
    m = _C.search(t or "")
    try:
        return float(m.group(1)) if m else 0.5
    except Exception:
        return 0.5


async def ask(c, sysm, um, temp):
    # robust: retry on ANY failure OR empty content (the prior run's
    # sc_clean collapse was empty-returns under concurrent load → parse-fail
    # → degenerate majority). Up to 6 tries with backoff.
    for attempt in range(6):
        try:
            r = await c.chat.completions.create(model=MODEL, temperature=temp,
                max_tokens=900,
                messages=[{"role": "system", "content": sysm},
                          {"role": "user", "content": um}])
            txt = r.choices[0].message.content
            if txt and txt.strip():
                return txt
            await asyncio.sleep(3)                # empty → retry
        except Exception as e:
            await asyncio.sleep(6 if ("429" in str(e) or "rate" in
                                      str(e).lower()) else 3)
    return ""


# semaphore is created INSIDE the running loop (module-level
# asyncio.Semaphore deadlocks: it binds to no/!current loop). v3 hung 43min
# on exactly this — fixed in v4.
_SEM = None
async def _one(c, sysm, um, temp):
    async with _SEM:
        return await ask(c, sysm, um, temp)
async def samples(c, sysm, um, n, temp):
    return await asyncio.gather(*[_one(c, sysm, um, temp) for _ in range(n)])


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(3)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Ns = [1, 3, 6, 9]
    out = {"model": MODEL, "n_problems": len(probs), "by_N": {}}
    for N in Ns:
        scm = scc = bom = 0
        cln_parse_fail = 0          # data-quality guard (sc_clean control)
        for p in probs:
            q, h, t = p["question"], p["misleading_hint"], p["answer"]
            mu = MisledSolverPrompts.format_user(q, h) + CONF
            cu = MathSolverPrompts.format_user(q)
            ms = await samples(c, MSYS, mu, N, 0.7)
            cs = await samples(c, CSYS, cu, N, 0.7)
            ma = [parse(x) for x in ms]; ca = [parse(x) for x in cs]
            mv = [x for x in ma if x is not None]
            cv = [x for x in ca if x is not None]
            if not cv:
                cln_parse_fail += 1   # all clean samples unparseable = artifact
            sc_m = Counter(mv).most_common(1)[0][0] if mv else None
            sc_c = Counter(cv).most_common(1)[0][0] if cv else None
            best = max(range(len(ms)), key=lambda i: conf(ms[i])) if ms else 0
            bo_m = parse(ms[best]) if ms else None
            scm += int(sc_m == t); scc += int(sc_c == t); bom += int(bo_m == t)
        n = len(probs)
        out["by_N"][N] = {"calls_per_problem": N,
            "sc_misled_acc": round(scm/n, 3),
            "sc_clean_acc": round(scc/n, 3),
            "bestof_misled_acc": round(bom/n, 3),
            "clean_allparsefail": cln_parse_fail,
            "data_quality_ok": cln_parse_fail <= 1}
        print(f"N={N} (B={N} calls): sc_misled={scm}/{n} "
              f"sc_clean={scc}/{n} bestof_misled={bom}/{n}")
    # matched GAIA point (same model, real budget ~6) from NX3-open
    import glob
    g = json.load(open(sorted(glob.glob(str(
        ROOT/"experiments/nx3_open/results/nx3open_*.json")))[-1]))
    out["GAIA_llama"] = {"acc": g["gaia"]["summary"]["accuracy"],
                         "approx_calls": 6, "note": "reuse NX3-open, same model"}
    json.dump(out, open(RES/f"c4_1_{ts}.json", "w"), indent=2)
    print(f"GAIA-llama @~6 calls = {out['GAIA_llama']['acc']:.0%}")
    print(f"Saved c4_1_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
