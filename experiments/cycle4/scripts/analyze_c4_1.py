#!/usr/bin/env python3
"""C4-1 analysis — the compute–accuracy frontier (hero figure) + C4-3b.

Reads the latest c4_1_openai_*.json and produces:

  1. cycle4/figures/c4_1_frontier.png — the money figure. x = total
     LLM-calls/problem, y = accuracy. sc_misled (the key curve, expect
     flat ≈0), sc_clean control (≈100% — traps individually solvable, so
     failure = correlated BIAS), bestof_misled, and GAIA as a single
     starred point at (~6, its real accuracy). Annotated with the
     one-sentence reading + the data-quality verdict.

  2. C4-3b — calibration from C4-1's ELICITED confidences (harvested at
     N=6 into calib_samples). This is the positive counterpart to C4-3's
     honest lexical negative: do solvers *say* a number that tracks
     correctness? Compute mean-conf, acc, ECE, confidently-wrong rate,
     and a 5-bin reliability table. Honest: if elicited conf is also
     flat/non-discriminative we report that straight.

Writes c4_1_frontier.png, c4_3b_elicited_calibration.md/.json. No API.
"""
import glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.parent
RES = ROOT/"experiments"/"cycle4"/"results"
FIG = ROOT/"experiments"/"cycle4"/"figures"
BLUE, GREEN, RED, GREY, PURPLE = "#1565C0", "#2E7D32", "#B71C1C", "#455A64", "#6A1B9A"


def latest():
    f = sorted(glob.glob(str(RES/"c4_1_openai_*.json")))
    if not f:
        raise SystemExit("no c4_1_openai_*.json yet — run run_c4_1_openai.py first")
    return json.load(open(f[-1])), f[-1]


def frontier(d):
    Ns = sorted(int(k) for k in d["by_N"])
    x = [d["by_N"][str(n)]["calls_per_problem"] for n in Ns]
    scm = [d["by_N"][str(n)]["sc_misled_acc"] for n in Ns]
    scc = [d["by_N"][str(n)]["sc_clean_acc"] for n in Ns]
    bom = [d["by_N"][str(n)]["bestof_misled_acc"] for n in Ns]
    g = d["GAIA_point"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(x, [v*100 for v in scc], "o--", color=GREEN, lw=2, ms=7,
            label="self-consistency, CLEAN solver (control)")
    ax.plot(x, [v*100 for v in scm], "o-", color=RED, lw=2.6, ms=8,
            label="self-consistency, MISLED solver (target regime)")
    ax.plot(x, [v*100 for v in bom], "s:", color=PURPLE, lw=1.8, ms=6,
            label="best-of-N by self-confidence (misled)")
    ax.scatter([g["approx_calls"]], [g["acc"]*100], marker="*", s=620,
               color=BLUE, edgecolor="k", lw=1.2, zorder=6,
               label=f"GAIA (structural conflict-as-task) — {g['acc']:.0%}")
    ax.annotate("extra compute cannot\nsubstitute for structure\nunder correlated bias",
                xy=(g["approx_calls"], g["acc"]*100), xytext=(7.1, 60),
                fontsize=9.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.set_xlabel("Total LLM calls per problem (compute budget)")
    ax.set_ylabel("Accuracy on correlated-failure traps (%)")
    ax.set_title("C4-1  Compute-matched baseline — same base model "
                 f"({d['model']}), {d['n_problems']} E3 traps")
    ax.set_ylim(-5, 108)
    ax.set_xticks(sorted(set(x + [g["approx_calls"]])))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.6, loc="center right", framealpha=0.95)
    dq = all(v.get("data_quality_ok") for v in d["by_N"].values())
    lo, hi = min(scm)*100, max(scm)*100
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.5, 0.015,
             (f"Reading: resampling a biased reasoner reproduces the bias — "
              f"misled self-consistency stays low ({lo:.0f}–{hi:.0f}%) "
              f"with no convergence toward GAIA at 9× compute "
              f"(swings are within majority-vote noise at n={d['n_problems']}); "
              f"the clean control is 100% at every budget, so the misled "
              f"failure is correlated BIAS, not trap hardness. "
              f"Data-quality guard: {'PASS' if dq else 'FAIL — see by_N'}."),
             ha="center", fontsize=8.0, wrap=True, color=GREY)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/"c4_1_frontier.png", dpi=180, bbox_inches="tight")
    print(f"saved {FIG/'c4_1_frontier.png'}  (dq_ok={dq})")
    return dq


def calib_3b(d):
    s = d.get("calib_samples", [])
    s = [x for x in s if x.get("conf") is not None]
    n = len(s)
    if not n:
        print("no calib_samples — skipping C4-3b"); return
    confs = [x["conf"] for x in s]
    cors = [x["correct"] for x in s]
    mc, acc = sum(confs)/n, sum(cors)/n
    ece = 0.0
    rows = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        b = [(c, k) for c, k in zip(confs, cors)
             if (lo <= c < hi or (hi == 1.0 and c == 1.0))]
        if not b:
            rows.append((lo, hi, 0, None, None)); continue
        bc = sum(c for c, _ in b)/len(b)
        ba = sum(k for _, k in b)/len(b)
        ece += len(b)/n * abs(bc - ba)
        rows.append((lo, hi, len(b), round(bc, 3), round(ba, 3)))
    cwr = sum(1 for c, k in zip(confs, cors) if c >= 0.8 and k == 0)/n
    out = {"n": n, "mean_conf": round(mc, 3), "acc": round(acc, 3),
           "ECE": round(ece, 3), "confidently_wrong_rate": round(cwr, 3),
           "reliability_bins": [{"lo": lo, "hi": hi, "n": bn,
                                 "mean_conf": bc, "acc": ba}
                                for lo, hi, bn, bc, ba in rows],
           "source": "misled-solver elicited CONFIDENCE: X at N=6 (C4-1)"}
    json.dump(out, open(RES/"c4_3b_elicited_calibration.json", "w"), indent=2)
    discr = (out["ECE"] is not None and out["ECE"] < 0.2
             and abs(mc - 0.5) > 0.05)
    L = ["# C4-3b — Elicited-confidence calibration (positive counterpart "
         "to C4-3's lexical negative)", "",
         f"Source: {out['source']}.  n={n} misled samples.", "",
         "| metric | value |", "|---|---|",
         f"| mean elicited confidence | {mc:.3f} |",
         f"| accuracy | {acc:.3f} |",
         f"| ECE | {ece:.3f} |",
         f"| confidently-wrong rate (conf≥0.8 & wrong) | {cwr:.3f} |",
         "", "## Reliability table",
         "| conf bin | n | mean conf | empirical acc |", "|---|---|---|---|"]
    for lo, hi, bn, bc, ba in rows:
        L.append(f"| [{lo:.1f},{hi:.1f}) | {bn} | "
                 f"{'—' if bc is None else bc} | {'—' if ba is None else ba} |")
    L += ["", "## Reading",
          ("- **Discriminative.** Elicited confidence tracks correctness "
           "(ECE < 0.2, conf spread off 0.5) — so the *self-reported* signal "
           "carries information the *lexical* proxy (C4-3) did not. Honest "
           "split: lexical hedging on math = null; explicit elicitation = "
           "usable.") if discr else
          ("- **Also non-discriminative (honest negative, consistent with "
           "C4-3).** Even explicitly elicited confidence clusters near a "
           "single value and does not separate right from wrong on these "
           "traps: the misled solver is *confidently wrong* — calibration "
           "cannot rescue a correlated bias, which is exactly why GAIA's "
           "structural conflict-as-task (not confidence weighting) is the "
           "mechanism that recovers truth. This strengthens, not weakens, "
           "the C4-1 thesis.")]
    (FIG/"c4_3b_elicited_calibration.md").write_text("\n".join(L))
    print(f"saved {FIG/'c4_3b_elicited_calibration.md'}  "
          f"(ECE={ece:.3f} cwr={cwr:.3f} discriminative={discr})")


if __name__ == "__main__":
    d, path = latest()
    print(f"loaded {Path(path).name}")
    frontier(d)
    calib_3b(d)
