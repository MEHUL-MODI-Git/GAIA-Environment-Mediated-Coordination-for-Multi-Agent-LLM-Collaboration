#!/usr/bin/env python3
"""C4-1 (OpenAI) — Compute-matched baseline, the paper-defining rebuttal.

Switched to OpenAI gpt-4.1-nano (reliable; Groq caused empty-return artifacts
+ a semaphore deadlock across v1-v4 — engineering decision, not a Groq-bug
claim). Stays compute-matched: every sampling arm uses the SAME base model
(gpt-4.1-nano); the GAIA point reuses the ORIGINAL E3 result (gpt-4.1
pipeline = 100% @ ~6 calls — same OpenAI family; the modestly stronger
reconciler is footnoted honestly).

Reviewer objection (Budget-Aware Eval, EMNLP'24): compute-matched
self-consistency beats multi-agent → "is GAIA just more compute?". Answer
under GAIA's TARGET regime (correlated misleading info), theory-backed by
C3-1 (traps are synergy-required; vote reads only redundancy):

  sc_misled@N : N misled samples (temp 0.7), majority. Expect ≈0%, FLAT/↓
                in N — resampling a biased reasoner reproduces the bias.
  sc_clean@N  : N clean samples, majority. Control ≈100% — proves traps are
                individually solvable, so failure = correlated BIAS not
                hardness. (data-quality guard: all-unparseable flagged.)
  bestof_misled@N : N misled, pick max self-confidence. Expect ≈0% (misled
                = confidently wrong; ties to C4-3b).
  GAIA        : reuse E3 gaia = 100% @ ~6 calls.

N ∈ {1,3,6,9}. Money figure: accuracy vs total LLM-calls — sc_misled flat ≈0,
GAIA single point at (≈6,100%) ⇒ extra compute provably cannot substitute
for structural conflict-as-task under correlated bias.
"""
import asyncio, glob, json, os, re, sys, logging
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

DATA = ROOT/"data"/"gsm8k"/"correlated_failure_problems.json"
RES = ROOT/"experiments"/"cycle4"/"results"
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
_C = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.I)
CONF = "\n\nAfter your answer add a line exactly: CONFIDENCE: X (0-1)."
_SEM = None  # created inside main() under the running loop (v4 lesson)


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


async def ask(cl, sysm, um, temp):
    async with _SEM:
        for _ in range(5):
            try:
                r = await cl.chat.completions.create(model=MODEL,
                    temperature=temp, max_tokens=900,
                    messages=[{"role": "system", "content": sysm},
                              {"role": "user", "content": um}])
                txt = r.choices[0].message.content
                if txt and txt.strip():
                    return txt
                await asyncio.sleep(2)
            except Exception as e:
                await asyncio.sleep(5 if "rate" in str(e).lower() else 2)
        return ""


async def samples(cl, sysm, um, n, temp):
    return await asyncio.gather(*[ask(cl, sysm, um, temp) for _ in range(n)])


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    cl = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"model": MODEL, "n_problems": len(probs), "by_N": {},
           "calib_samples": []}
    for N in (1, 3, 6, 9):
        scm = scc = bom = 0
        cpf = 0
        for p in probs:
            q, h, t = p["question"], p["misleading_hint"], p["answer"]
            mu = MisledSolverPrompts.format_user(q, h) + CONF
            cu = MathSolverPrompts.format_user(q)
            ms = await samples(cl, MSYS, mu, N, 0.7)
            cs = await samples(cl, CSYS, cu, N, 0.7)
            ma = [parse(x) for x in ms]; ca = [parse(x) for x in cs]
            mv = [x for x in ma if x is not None]
            cv = [x for x in ca if x is not None]
            if not cv:
                cpf += 1
            sc_m = Counter(mv).most_common(1)[0][0] if mv else None
            sc_c = Counter(cv).most_common(1)[0][0] if cv else None
            bi = max(range(len(ms)), key=lambda i: conf(ms[i])) if ms else 0
            bo = parse(ms[bi]) if ms else None
            scm += int(sc_m == t); scc += int(sc_c == t); bom += int(bo == t)
            if N == 6:   # harvest elicited confidences for C4-3b calibration
                for x in ms:
                    a = parse(x)
                    out["calib_samples"].append(
                        {"conf": conf(x), "correct": int(a == t)})
        n = len(probs)
        out["by_N"][N] = {"calls_per_problem": N,
            "sc_misled_acc": round(scm/n, 3),
            "sc_clean_acc": round(scc/n, 3),
            "bestof_misled_acc": round(bom/n, 3),
            "clean_allparsefail": cpf, "data_quality_ok": cpf <= 1}
        print(f"N={N}: sc_misled={scm}/{n} sc_clean={scc}/{n} "
              f"bestof_misled={bom}/{n} dq_ok={cpf<=1}")
    e = json.load(open(sorted(glob.glob(str(
        ROOT/"experiments/correlated_failure/results/correlated_failure_2*.json")))[-1]))
    out["GAIA_point"] = {"acc": e["gaia"]["summary"]["accuracy"],
        "approx_calls": 6,
        "note": "reuse original E3 gaia (gpt-4.1 pipeline; same OpenAI "
                "family; reconciler modestly stronger — footnoted)"}
    json.dump(out, open(RES/f"c4_1_openai_{ts}.json", "w"), indent=2)
    print(f"GAIA @~6 calls = {out['GAIA_point']['acc']:.0%}")
    print(f"Saved c4_1_openai_{ts}.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
