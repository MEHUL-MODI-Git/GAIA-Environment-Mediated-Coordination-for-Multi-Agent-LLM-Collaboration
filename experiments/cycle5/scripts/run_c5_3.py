#!/usr/bin/env python3
"""C5-3 — Information-controlled architecture isolation + Murphy decomposition.

Defends "it's the ARCHITECTURE, not the information" against the Ao et al.
(2026) non-identifiability result, using the 2605.03310 protocol:

  FIXED  : one model (gpt-4.1-nano); identical 13-trap set; identical
           EXOGENOUS information available to the system = {question, hint};
           identical per-call output cap (max_tokens). No tools, no retrieval,
           no web — nothing can smuggle information between arms.
  VARIES : ONLY the coordination/role-prompt block. Total compute is
           ENDOGENOUS (recorded, not capped) — different structures naturally
           spend different amounts; capping would break architectural
           integrity (their argument).

Arms (all receive exactly {Q, H}; they DISTRIBUTE it differently — that
distribution IS the architecture, not an information difference):
  gaia_conflict_bb : 2 misled(get H) + 1 clean(no H) → typed CONFLICT →
                     error-diagnosis reconciler.
  roundrobin_debate: 3 agents (all see Q,H), 2 rounds seeing peers' latest,
                     then majority.
  broadcast_bb     : same 3 chains as GAIA, but a PLAIN aggregator (signals
                     disabled — no CONFLICT-triggered error-analysis; just
                     "synthesize an answer") ≈ the 2510.01285 design.

Each arm emits a 0–1 confidence. Analysis: Brier score + Murphy
decomposition BS = reliability − resolution + uncertainty (K=5 bins) — a
NEW analysis representation for this corpus. Audit artifact: every arm's
rendered system prompts dumped; only the role/coordination block differs;
exogenous info identical. Honest: n=13 ⇒ coarse bins, stated; if debate
matches GAIA accuracy but differs in resolution we report that.

Also dumps per-round messages → consumed by C5-4 (semantic-drift) for free.
OpenAI; robust pattern.
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
RES = ROOT/"experiments"/"cycle5"/"results"
MODEL = "gpt-4.1-nano"
CSYS, MSYS = MathSolverPrompts.SYSTEM, MisledSolverPrompts.SYSTEM
RP = MathReconcilerPrompts()
_I = re.compile(r"\*\*\s*Final Answer:\s*\[?(-?\d[\d,]*)\]?\s*\*\*", re.I)
_C = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.I)
CONF = "\n\nAfter the final answer add a line exactly: CONFIDENCE: X (0-1)."
DEBATE_SYS = ("You are a careful mathematician in a 3-person panel. You see "
              "the problem, an external procedural hint, and (after round 1) "
              "your peers' latest answers. Reason independently; revise only "
              "if genuinely convinced. End with **Final Answer: N** then "
              "CONFIDENCE: X.")
AGG_SYS = ("You are an aggregator. You are given a problem and three "
           "independent solution attempts. Synthesize the single best answer. "
           "End with **Final Answer: N** then CONFIDENCE: X.")
_SEM = None


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
        return min(1.0, max(0.0, float(m.group(1)))) if m else 0.5
    except Exception:
        return 0.5


async def ask(c, s, u, t=0.0):
    async with _SEM:
        for _ in range(5):
            try:
                r = await c.chat.completions.create(model=MODEL,
                    temperature=t, max_tokens=900,
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


async def arm_gaia(c, p):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    (m0, k0), (m1, k1), (cl, kc) = await asyncio.gather(
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h) + CONF),
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h) + CONF),
        ask(c, CSYS, MathSolverPrompts.format_user(q) + CONF))
    tok = k0+k1+kc
    an = {"S1": parse(m0), "S2": parse(m1), "S3": parse(cl)}
    msgs = [m0, m1, cl]
    vals = [v for v in an.values() if v is not None]
    if len(set(vals)) >= 2:
        summ = ", ".join(f"{k}={an[k]}" for k in an)
        rr, kr = await ask(c, RP.SYSTEM, RP.format_user(
            q, summ, [("S1", m0), ("S2", m1), ("S3", cl)]) + CONF)
        tok += kr; msgs.append(rr)
        pred, cf = parse(rr), conf(rr)
    else:
        pred = vals[0] if vals else None
        cf = max(conf(m0), conf(m1), conf(cl))
    return int(pred == t), cf, tok, msgs


async def arm_debate(c, p, rounds=2):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    base = (f"Problem: {q}\n\nExternal procedural hint (may or may not be "
            f"reliable): {h}\n")
    hist = [[], [], []]
    last = [""]*3
    tok = 0
    rnd_msgs = []
    for r in range(rounds):
        peer = ""
        if r > 0:
            peer = "\n\nPeers' latest answers: " + "; ".join(
                f"A{i+1}={parse(last[i])}" for i in range(3))
        outs = await asyncio.gather(*[
            ask(c, DEBATE_SYS, base + peer + CONF) for _ in range(3)])
        for i, (txt, k) in enumerate(outs):
            last[i] = txt; hist[i].append(txt); tok += k
        rnd_msgs.append(list(last))
    finals = [parse(x) for x in last]
    vals = [v for v in finals if v is not None]
    pred = Counter(vals).most_common(1)[0][0] if vals else None
    cf = sum(conf(x) for x in last)/3
    return int(pred == t), cf, tok, rnd_msgs


async def arm_broadcast(c, p):
    q, h, t = p["question"], p["misleading_hint"], p["answer"]
    (m0, k0), (m1, k1), (cl, kc) = await asyncio.gather(
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h) + CONF),
        ask(c, MSYS, MisledSolverPrompts.format_user(q, h) + CONF),
        ask(c, CSYS, MathSolverPrompts.format_user(q) + CONF))
    tok = k0+k1+kc
    u = (f"Problem: {q}\n\nThree independent attempts:\n"
         f"=== A1 ===\n{m0}\n=== A2 ===\n{m1}\n=== A3 ===\n{cl}\n")
    rr, kr = await ask(c, AGG_SYS, u + CONF)
    tok += kr
    return int(parse(rr) == t), conf(rr), tok, [m0, m1, cl, rr]


def murphy(pairs, K=5):
    """BS = reliability - resolution + uncertainty (Murphy 1973)."""
    n = len(pairs)
    if not n:
        return {}
    obar = sum(o for _, o in pairs)/n
    unc = obar*(1-obar)
    rel = res = 0.0
    bs = sum((f-o)**2 for f, o in pairs)/n
    for b in range(K):
        lo, hi = b/K, (b+1)/K
        bin_ = [(f, o) for f, o in pairs
                if (lo <= f < hi or (b == K-1 and f == 1.0))]
        if not bin_:
            continue
        nb = len(bin_)
        fbar = sum(f for f, _ in bin_)/nb
        obark = sum(o for _, o in bin_)/nb
        rel += nb*(fbar-obark)**2
        res += nb*(obark-obar)**2
    return {"brier": round(bs, 4), "reliability": round(rel/n, 4),
            "resolution": round(res/n, 4), "uncertainty": round(unc, 4),
            "decomp_check": round(rel/n - res/n + unc, 4)}


async def main():
    global _SEM
    _SEM = asyncio.Semaphore(5)
    RES.mkdir(parents=True, exist_ok=True)
    probs = json.load(open(DATA))
    c = openai.AsyncOpenAI()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = len(probs)
    arms = {"gaia_conflict_bb": arm_gaia,
            "roundrobin_debate": arm_debate,
            "broadcast_bb": arm_broadcast}
    out = {"model": MODEL, "n": n,
           "info_control": ("exogenous info fixed = {question, hint}; one "
               "model; output cap 900 tok/call; no tools/retrieval/web; "
               "ONLY the coordination/role-prompt block varies; total "
               "compute endogenous (recorded). Audit: prompts dumped."),
           "arms": {}}
    transcripts = {}                       # for C5-4 semantic-drift (free)
    for name, fn in arms.items():
        ok = 0; pairs = []; toks = []; tr = []
        for p in probs:
            try:
                c_ok, cf, tk, msgs = await asyncio.wait_for(
                    fn(c, p), timeout=200)
            except Exception:
                c_ok, cf, tk, msgs = 0, 0.5, 0, []
            ok += c_ok; pairs.append((cf, c_ok)); toks.append(tk)
            tr.append({"pid": p["problem_id"], "answer": p["answer"],
                       "messages": msgs})
        sv = [x for x in toks if x > 0]
        out["arms"][name] = {"accuracy": round(ok/n, 3),
            "mean_endogenous_tokens": round(sum(sv)/len(sv), 1) if sv else 0,
            "murphy": murphy(pairs)}
        transcripts[name] = tr
        m = out["arms"][name]
        print(f"[{name:18s}] acc={m['accuracy']:.0%} "
              f"tok≈{m['mean_endogenous_tokens']:.0f} "
              f"Brier={m['murphy'].get('brier')} "
              f"resol={m['murphy'].get('resolution')}")
    # prompt-diff audit artifact
    audit = {"shared_exogenous_info": "{question, hint} only; no tools/web",
             "role_blocks": {
                 "gaia_conflict_bb": [MSYS[:300], CSYS[:300], RP.SYSTEM[:300]],
                 "roundrobin_debate": [DEBATE_SYS],
                 "broadcast_bb": [MSYS[:300], CSYS[:300], AGG_SYS]},
             "note": ("only the role/coordination instruction blocks differ; "
                      "the problem text + hint (the exogenous information) and "
                      "the model and the 900-token output cap are identical "
                      "across all arms")}
    json.dump(out, open(RES/f"c5_3_{ts}.json", "w"), indent=2)
    json.dump(audit, open(RES/f"c5_3_prompt_audit_{ts}.json", "w"), indent=2)
    json.dump(transcripts,
              open(RES/f"c5_3_transcripts_{ts}.json", "w"), indent=2)
    print(f"Saved c5_3_{ts}.json (+ prompt_audit, transcripts for C5-4)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
