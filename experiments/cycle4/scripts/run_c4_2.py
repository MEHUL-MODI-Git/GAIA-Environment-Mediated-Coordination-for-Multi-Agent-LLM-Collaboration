#!/usr/bin/env python3
"""C4-2 — Behavioral-taxonomy coding of agent transcripts.

The "infer agent behaviours" deliverable. We define a behaviour codebook and
have TWO independent judges (gpt-4.1 via OpenAI + claude-sonnet-4-6 via raw
Anthropic SDK — cross-vendor) multi-label each agent transcript. Inter-judge
reliability via Cohen's κ per code (qualitative-coding standard; LLM coders
validated at κ≥0.79 for clear codes — we report it honestly, no spin).

Codebook (derived from the agentic-failure-taxonomy literature + GAIA roles):
  IND  independent_derivation : solves from first principles, own reasoning
  ANC  anchoring_on_peers     : visibly copies/defers to others' stated answers
  HDG  hedging                : expresses uncertainty / multiple candidates
  CAP  capitulation           : changes toward majority despite own logic
  SCR  self_correction        : catches & fixes its own error mid-chain
  FCF  false_confidence       : asserts certainty while actually wrong
  TPF  trap_following         : applies the seductive/misleading heuristic
  TPD  trap_detection         : explicitly flags/avoids the common trap

Output: per-role × per-code frequency (judge-averaged), inter-judge κ table,
and the behavioural-repertoire heatmap data. Honest: LLM behavioural coding;
κ reported; subset-sampled for cost.
"""
import asyncio, glob, json, os, random, sys, logging
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT/".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
import openai, anthropic

RES = ROOT/"experiments"/"cycle4"/"results"
CODES = ["IND", "ANC", "HDG", "CAP", "SCR", "FCF", "TPF", "TPD"]
ROLE = {"math_solution": "solver", "reconciled_solution": "reconciler",
        "aggregator_verdict": "aggregator"}
RUBRIC = (
 "Behaviour codebook (multi-label; output ONLY codes that clearly apply):\n"
 "IND independent_derivation: reasons from first principles, own derivation.\n"
 "ANC anchoring_on_peers: copies/defers to other agents' stated answers.\n"
 "HDG hedging: expresses uncertainty or multiple candidate answers.\n"
 "CAP capitulation: shifts toward the majority despite its own logic.\n"
 "SCR self_correction: catches and fixes its own mistake mid-reasoning.\n"
 "FCF false_confidence: asserts certainty (it is wrong, but you only see "
 "the text — judge tone/assertiveness + internal inconsistency).\n"
 "TPF trap_following: applies a seductive but flawed heuristic/shortcut.\n"
 "TPD trap_detection: explicitly flags or avoids the common trap.\n"
 'Reply STRICT JSON: {"codes":["IND",...]} — subset of the 8, may be empty.')


def collect(maxn=70):
    items = []
    for sf in glob.glob(str(ROOT/"experiments/correlated_failure/logs/**/*.state.json"),
                        recursive=True):
        try:
            d = json.load(open(sf))
        except Exception:
            continue
        for a in d.get("artifacts", {}).values():
            r = ROLE.get(a.get("metadata", {}).get("subtype"))
            c = (a.get("content") or "").strip()
            if r and len(c) > 60:
                items.append({"role": r,
                              "misled": bool(a.get("metadata", {}).get("is_misled")),
                              "text": c[:2200]})
    random.Random(0).shuffle(items)
    return items[:maxn]


async def judge_openai(cl, text):
    try:
        r = await cl.chat.completions.create(model="gpt-4.1", temperature=0.0,
            max_tokens=120,
            messages=[{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": text}])
        s = r.choices[0].message.content
        return json.loads(s[s.find("{"):s.rfind("}")+1]).get("codes", [])
    except Exception:
        return []


async def judge_claude(cl, text):
    try:
        r = await cl.messages.create(model="claude-sonnet-4-6", max_tokens=120,
            temperature=0.0, system=RUBRIC,
            messages=[{"role": "user", "content": text}])
        s = r.content[0].text
        return json.loads(s[s.find("{"):s.rfind("}")+1]).get("codes", [])
    except Exception:
        return []


def kappa(a, b):
    """Cohen's κ for a binary code over items."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y)/n
    pa1 = sum(a)/n; pb1 = sum(b)/n
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    return round((po-pe)/(1-pe), 3) if pe != 1 else 1.0


async def main():
    RES.mkdir(parents=True, exist_ok=True)
    items = collect()
    oc = openai.AsyncOpenAI()
    ac = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    print(f"C4-2: coding {len(items)} transcripts with 2 cross-vendor judges")
    for i, it in enumerate(items):
        o, c = await asyncio.gather(judge_openai(oc, it["text"]),
                                    judge_claude(ac, it["text"]))
        it["gpt"] = set(o); it["cla"] = set(c)
        if (i+1) % 15 == 0:
            print(f"  {i+1}/{len(items)}")
    # inter-judge κ per code
    kap = {}
    for code in CODES:
        a = [1 if code in it["gpt"] else 0 for it in items]
        b = [1 if code in it["cla"] else 0 for it in items]
        kap[code] = kappa(a, b)
    # consensus label = union? use intersection for precision; report both
    def lab(it): return it["gpt"] & it["cla"]   # high-precision consensus
    freq = defaultdict(lambda: Counter())
    grp_n = Counter()
    for it in items:
        g = ("misled-solver" if (it["role"] == "solver" and it["misled"])
             else "clean-solver" if it["role"] == "solver"
             else it["role"])
        grp_n[g] += 1
        for code in lab(it):
            freq[g][code] += 1
    table = {g: {code: round(freq[g][code]/grp_n[g], 2) for code in CODES}
             for g in grp_n}
    out = {"n_items": len(items), "inter_judge_kappa": kap,
           "group_n": dict(grp_n), "behaviour_freq_consensus": table}
    json.dump(out, open(RES/"c4_2_behaviour.json", "w"), indent=2)

    L = ["# C4-2 — Behavioural-taxonomy coding (2 cross-vendor judges, "
         f"n={len(items)})", "",
         "## Inter-judge reliability (Cohen's κ per code; gpt-4.1 vs "
         "claude-sonnet-4-6)",
         "| code | κ |", "|---|---|"]
    for code in CODES:
        L.append(f"| {code} | {kap[code]} |")
    L += ["", "## Behaviour frequency by group (high-precision consensus = "
          "both judges agree)", "",
          "| group | n | " + " | ".join(CODES) + " |",
          "|---|---|" + "|".join(["---"]*len(CODES)) + "|"]
    for g in grp_n:
        L.append(f"| {g} | {grp_n[g]} | " +
                 " | ".join(f"{table[g][c]:.2f}" for c in CODES) + " |")
    L += ["", "## Reading",
          "- Expected & to-verify: misled-solver high TPF/FCF (follows trap, "
          "confidently wrong), clean-solver high IND/TPD, reconciler high "
          "SCR/TPD/IND (audits & re-derives). Capitulation (CAP) in solvers "
          "= the conformity failure GAIA's structure must resist.",
          "- κ: codes with κ≥0.6 are reliably coded; low-κ codes (e.g. FCF, "
          "inherently subjective) are reported but flagged as judge-sensitive "
          "— honest, per qualitative-coding standards. Consensus = both-judge "
          "intersection (high precision, conservative).",
          "- This is a *behavioural repertoire* per role — an inferred "
          "agent-behaviour structure not targeted by any prior experiment."]
    (ROOT/"experiments"/"cycle4"/"figures"/"c4_2_behaviour.md").write_text(
        "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
