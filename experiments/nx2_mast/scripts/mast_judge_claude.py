#!/usr/bin/env python3
"""NX2 cross-VENDOR judge — Claude judging OpenAI-family agent traces.

Closes the documented limitation in DEEP_ANALYSIS §6: the prior calibration
was cross-FAMILY but same-VENDOR (gpt-4.1 vs gpt-4o). This re-classifies the
SAME shuffled baseline_weak failed traces with a **Claude** judge (raw
anthropic SDK — the GAIA AnthropicLLM wrapper has a bug, bypassed on purpose)
so judge_agreement.py can report a true cross-VENDOR Cohen's κ + the binary
headline robustness. Self-enhancement bias (GPT judging GPT) is structurally
removed here.

Reuses the EXACT rubric + trace summarizers from mast_classifier.py so the
only changed variable is the judge model/vendor.
"""
import asyncio, json, os, random, sys, logging
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

import mast_classifier as mc          # reuse SYS, summarizers
import anthropic

JUDGE_MODEL = "claude-sonnet-4-6"     # verified working via raw SDK
RES = Path(__file__).parent.parent / "results"


async def classify(client, trace):
    try:
        r = await client.messages.create(
            model=JUDGE_MODEL, max_tokens=300, temperature=0.0,
            system=mc.SYS,
            messages=[{"role": "user", "content": trace}])
        raw = r.content[0].text.strip()
        a, b = raw.find("{"), raw.rfind("}")
        return json.loads(raw[a:b+1])
    except Exception as e:
        return {"modes": ["FM-X.0"], "primary": "FM-X.0",
                "confidence": 0.0, "evidence": f"err:{str(e)[:40]}"}


async def main():
    base = sorted((ROOT/"experiments/correlated_failure/results").glob(
        "correlated_failure_2*.json"))[-1]
    nx1 = sorted(f for f in (ROOT/"experiments/nx1_baselines/results").glob(
        "nx1_2*.json") if "checkpoint" not in f.name)[-1]

    traces = []
    for rp in (base, nx1):
        d = json.load(open(rp))
        for cond, v in d.items():
            for r in v.get("results", []):
                t = mc.summarize_result_record(r)
                if t:
                    traces.append((f"{cond}:{r.get('problem_id') or r.get('puzzle_id')}", t))
    random.Random(0).shuffle(traces)          # SAME seed/order as OpenAI judges
    traces = traces[:80]
    print(f"[baseline_weak_claude] {len(traces)} traces, judge={JUDGE_MODEL}")

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out, prim = [], Counter()
    for i, (tid, tr) in enumerate(traces, 1):
        res = await classify(client, tr)
        out.append({"trace": tid, **res})
        prim[res.get("primary", "FM-X.0")] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(traces)}")
    RES.mkdir(parents=True, exist_ok=True)
    json.dump({"label": "baseline_weak_claude", "judge_model": JUDGE_MODEL,
               "n_failed_traces": len(traces),
               "primary_counts": dict(prim.most_common()),
               "per_trace": out},
              open(RES/"mast_baseline_weak_claude.json", "w"), indent=2)
    print(f"primary modes (Claude): {dict(prim.most_common(8))}")
    print(f"Saved {RES/'mast_baseline_weak_claude.json'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
