#!/usr/bin/env python3
"""NX2 — MAST failure-mode classifier (the paper's intellectual centerpiece).

Maps every FAILED trace to the MAST taxonomy (Cemri et al., Berkeley,
ICLR/NeurIPS 2025, arXiv 2503.13657): 14 failure modes in 3 categories. By
classifying GAIA's failures vs a baseline's failures we show *which* MAST
modes GAIA's structure removes — closing MAST's own open loop (they showed
tactical fixes are insufficient and called for structural ones; GAIA is the
structural one).

Bias mitigations baked in (from the LLM-as-judge literature):
  * rubric authored directly from the 14 official MAST definitions
  * judge is BLIND to which system produced the trace (condition stripped)
  * forced structured JSON output (no free-form preference)
  * traces shuffled; one mode-set per trace + confidence
  * judge model family flagged (only OpenAI configured here — we use gpt-4.1
    as judge while agents used gpt-4.1-nano/mini, a partial cross-tier
    separation; full cross-family (Claude) judge is noted as the rigorous
    upgrade and left as a CLI swap).

Input: any *.state.json dumps (full reasoning chains) OR results JSON with
per-problem records. We classify only FAILED episodes.

Usage:
  python experiments/nx2_mast/scripts/mast_classifier.py \
     --glob 'experiments/**/logs/**/*.state.json' --label GAIA
"""
import argparse, asyncio, glob, json, os, sys, logging
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
_env = ROOT / ".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier

JUDGE_MODEL = "gpt-4.1"  # stronger than the agent models (gpt-4.1-nano/mini)

MAST = """\
MAST taxonomy — 14 failure modes (Cemri et al. 2025). Pick ALL that the trace
exhibits; if the episode failed but no mode clearly applies, use FM-X.0.

FC1 — Specification & System Design:
 FM-1.1 Disobey task specification: ignored stated constraints/requirements.
 FM-1.2 Disobey role specification: an agent acted outside its assigned role.
 FM-1.3 Step repetition: needlessly redid completed steps.
 FM-1.4 Loss of conversation history: context truncated; reverted to old state.
 FM-1.5 Unaware of termination conditions: didn't recognize stop criteria.
FC2 — Inter-Agent Misalignment:
 FM-2.1 Conversation reset: unwarranted restart losing progress.
 FM-2.2 Fail to ask clarification: proceeded under ambiguity without asking.
 FM-2.3 Task derailment: drifted from the intended objective.
 FM-2.4 Information withholding: an agent failed to share decisive info.
 FM-2.5 Ignored other agent's input: disregarded a correct peer contribution.
 FM-2.6 Reasoning-action mismatch: final action contradicts its own reasoning.
FC3 — Verification & Termination:
 FM-3.1 Premature termination: ended before the objective was met.
 FM-3.2 No/incomplete verification: skipped checking; error propagated.
 FM-3.3 Incorrect verification: verified but the check itself was wrong.
FM-X.0 Other/unclear: failed but none of the above clearly applies.
"""

SYS = (f"You are a rigorous failure-mode annotator. You will be shown an "
       f"anonymized multi-agent solving trace that ENDED IN FAILURE (wrong "
       f"final answer). Classify it strictly using this taxonomy.\n\n{MAST}\n"
       f"Do NOT speculate about which framework produced it. Judge only what "
       f"the trace shows. Output STRICT JSON: "
       f'{{"modes":["FM-x.y",...],"primary":"FM-x.y","confidence":0-1,'
       f'"evidence":"<=40 words"}}.')


def summarize_state(sf):
    """Compress a state dump into a judge-readable, condition-anonymized trace."""
    try:
        d = json.load(open(sf))
    except Exception:
        return None
    extra = d.get("extra", {})
    if extra.get("passed") is True:
        return None  # only classify failures
    arts = list(d.get("artifacts", {}).values())
    sigs = list(d.get("signals", {}).values())
    parts = [f"Episode: {d.get('episode_id','?')}",
             f"Outcome: FAILED. Ground truth withheld."]
    # agents + their (truncated) reasoning, role-anonymized
    for i, a in enumerate(sorted(arts, key=lambda x: x.get("created_at", ""))):
        md = a.get("metadata", {})
        sub = md.get("subtype", "?")
        content = (a.get("content", "") or "")[:700]
        parts.append(f"[Step {i} | type={sub}] {content}")
    for s in sigs:
        parts.append(f"[SIGNAL {s.get('type')}] {(s.get('description') or '')[:160]}")
    if not arts:
        return None
    return "\n".join(parts)[:9000]


def summarize_result_record(r):
    """Fallback for results-JSON records without a state dump."""
    if r.get("passed"):
        return None
    bits = [f"Problem: {r.get('problem_id') or r.get('puzzle_id')}",
            "Outcome: FAILED.",
            f"Proposed: {r.get('proposed_answer') or r.get('proposed_solution')}"]
    for k in ("misled_answers", "clean_answer", "majority_answer",
              "round_answers", "trust_scores", "n_contradictions_found"):
        if k in r and r[k] is not None:
            bits.append(f"{k}: {r[k]}")
    return "\n".join(str(b) for b in bits)[:4000]


async def classify(judge, trace):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": trace}]
    raw = await judge.call_llm(msgs, temperature=0.0) if hasattr(judge, "call_llm") \
        else (await judge.agenerate(msgs, temperature=0.0)).content
    raw = raw.strip()
    a, b = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[a:b + 1])
    except Exception:
        return {"modes": ["FM-X.0"], "primary": "FM-X.0",
                "confidence": 0.0, "evidence": "parse_failed"}


class _JudgeWrap:
    def __init__(self, llm): self.llm = llm
    async def call_llm(self, msgs, **kw):
        return (await self.llm.agenerate(msgs, **kw)).content


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True,
                    help="glob(s) for *.state.json (repeatable)")
    ap.add_argument("--results", action="append", default=[],
                    help="results JSON path(s); classifies failed records")
    ap.add_argument("--label", required=True, help="system label, e.g. GAIA or debate")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    args = ap.parse_args()

    judge = _JudgeWrap(OpenAILLM(model=args.judge_model, tier=ModelTier.SLOW))

    traces = []
    for g in args.glob:
        for sf in glob.glob(str(ROOT / g), recursive=True):
            t = summarize_state(sf)
            if t:
                traces.append((Path(sf).stem, t))
    for rp in args.results:
        d = json.load(open(rp))
        for cond, v in d.items():
            for r in v.get("results", []):
                t = summarize_result_record(r)
                if t:
                    traces.append((f"{cond}:{r.get('problem_id') or r.get('puzzle_id')}", t))

    import random
    random.Random(0).shuffle(traces)            # kill order bias
    traces = traces[:args.max]
    print(f"[{args.label}] classifying {len(traces)} FAILED traces "
          f"(judge={args.judge_model})")

    out, mode_counts, primary_counts = [], Counter(), Counter()
    for i, (tid, tr) in enumerate(traces, 1):
        res = await classify(judge, tr)
        out.append({"trace": tid, **res})
        for m in res.get("modes", []):
            mode_counts[m] += 1
        primary_counts[res.get("primary", "FM-X.0")] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(traces)}")

    od = Path(__file__).parent.parent / "results"
    od.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": args.label, "judge_model": args.judge_model,
        "n_failed_traces": len(traces),
        "mode_counts": dict(mode_counts.most_common()),
        "primary_counts": dict(primary_counts.most_common()),
        "per_trace": out,
    }
    op = od / f"mast_{args.label}.json"
    json.dump(summary, open(op, "w"), indent=2)
    print(f"\n[{args.label}] primary failure modes: "
          f"{dict(primary_counts.most_common(8))}")
    print(f"Saved {op}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
