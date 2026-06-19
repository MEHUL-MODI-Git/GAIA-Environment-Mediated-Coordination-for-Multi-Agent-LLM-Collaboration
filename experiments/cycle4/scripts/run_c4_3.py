#!/usr/bin/env python3
"""C4-3 — Calibration / reliability of agent reasoning (free, lexical).

Question: do agents' *expressed* confidence track correctness, and is the
reconciler better-calibrated than solvers? The dangerous failure mode is
*confidently wrong* (high assertiveness, incorrect) — especially misled
solvers. We derive a lexical confidence proxy from each agent's transcript
(assertion markers − hedge markers, normalized) and bin vs correctness to
build a reliability curve + Expected Calibration Error (ECE).

Free: mines existing E3 state dumps (per-agent content + answer + truth).
Honest: lexical assertiveness is a *proxy* for confidence (no model logits);
reported as such. Still a standard, rigorous calibration lens not yet applied.
"""
import glob, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
OUT = ROOT/"experiments"/"cycle4"
ASSERT = re.compile(r"\b(clearly|obviously|certainly|definitely|must be|"
                    r"exactly|precisely|the answer is|therefore the|hence the|"
                    r"without doubt|undoubtedly|simply)\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|might|could be|possibly|i think|"
                   r"not sure|unclear|approximately|roughly|seems|likely|"
                   r"or it could|alternatively|uncertain)\b", re.I)
ROLE = {"math_solution": "solver", "reconciled_solution": "reconciler",
        "aggregator_verdict": "aggregator"}


def conf_proxy(text):
    a = len(ASSERT.findall(text)); h = len(HEDGE.findall(text))
    # squash to 0..1: more assertions vs hedges → higher confidence
    return 1/(1+pow(2.718, -(a-h)/2.0))


def ece(points, bins=5):
    """points: list of (conf, correct). Expected Calibration Error."""
    if not points:
        return None
    e = 0.0; n = len(points)
    for b in range(bins):
        lo, hi = b/bins, (b+1)/bins
        bp = [(c, y) for c, y in points if (lo <= c < hi or (b == bins-1 and c == 1.0))]
        if not bp:
            continue
        conf = sum(c for c, _ in bp)/len(bp)
        acc = sum(y for _, y in bp)/len(bp)
        e += (len(bp)/n)*abs(conf-acc)
    return round(e, 3)


def main():
    pts = defaultdict(list)   # role/group -> [(conf, correct)]
    for sf in glob.glob(str(ROOT/"experiments/correlated_failure/logs/**/*.state.json"),
                        recursive=True):
        try:
            d = json.load(open(sf))
        except Exception:
            continue
        ex = d.get("extra", {}); truth = ex.get("ground_truth")
        for a in d.get("artifacts", {}).values():
            md = a.get("metadata", {})
            r = ROLE.get(md.get("subtype"))
            if not r:
                continue
            txt = (a.get("content") or "")
            if len(txt) < 40 or truth is None:
                continue
            ans = md.get("answer")
            correct = 1 if (ans is not None and ans == truth) else 0
            grp = ("misled-solver" if (r == "solver" and md.get("is_misled"))
                   else "clean-solver" if r == "solver" else r)
            pts[grp].append((conf_proxy(txt), correct))

    L = ["# C4-3 — Calibration / reliability (lexical proxy, free, E3 dumps)",
         "",
         "| group | n | mean confidence | accuracy | **ECE** | "
         "confidently-wrong rate |", "|---|---|---|---|---|---|"]
    res = {}
    for g, P in sorted(pts.items()):
        if len(P) < 5:
            continue
        mc = sum(c for c, _ in P)/len(P)
        acc = sum(y for _, y in P)/len(P)
        cw = sum(1 for c, y in P if c >= 0.6 and y == 0)/len(P)  # high-conf wrong
        e = ece(P)
        res[g] = {"n": len(P), "mean_conf": round(mc, 3),
                  "acc": round(acc, 3), "ECE": e,
                  "confidently_wrong_rate": round(cw, 3)}
        L.append(f"| {g} | {len(P)} | {mc:.3f} | {acc:.3f} | **{e}** | "
                 f"{cw:.0%} |")
    L += ["", "## Reading",
          "- **Confidently-wrong rate** (assertive AND incorrect) is the "
          "dangerous mode. Expect misled-solver high (it follows the trap "
          "with conviction) — this is *why* majority/self-consistency fail "
          "(they aggregate confident-but-wrong agents) and why GAIA's "
          "reconciler (which audits reasoning, not confidence) is needed.",
          "- **ECE**: lower = better-calibrated. Hypothesis: reconciler ECE "
          "< misled-solver ECE (audit reduces over-confidence).",
          "- Honest: assertiveness is a lexical proxy for confidence (no "
          "logits); the *relative* ordering across roles is the claim, not "
          "absolute calibration. Free re-analysis, fully reproducible."]
    (OUT/"figures"/"c4_3_calibration.md").write_text("\n".join(L))
    json.dump(res, open(OUT/"results"/"c4_3_calibration.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
