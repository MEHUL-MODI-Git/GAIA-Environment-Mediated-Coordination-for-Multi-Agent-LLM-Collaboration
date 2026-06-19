#!/usr/bin/env python3
"""C5-5 figure — reconciler information-bottleneck (honest: saturates at L0)."""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent.parent/"results"
FIG = Path(__file__).parent.parent/"figures"
BLUE, GREY, RED = "#1565C0", "#455A64", "#B71C1C"

d = json.load(open(sorted(glob.glob(str(RES/"c5_5_*.json")))[-1]))
L = ["L0_answers_only", "L1_final_lines", "L2_full_chains", "L3_conflict_sig"]
lab = ["L0\nanswers only\n(chains withheld)", "L1\n+final lines",
       "L2\n+full chains\n(standard GAIA)", "L3\n+explicit\nCONFLICT signal"]
acc = [d["by_level"][k]["accuracy"]*100 for k in L]

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.plot(range(4), acc, "o-", color=BLUE, lw=2.6, ms=10)
for i, a in enumerate(acc):
    ax.annotate(f"{a:.0f}%", (i, a), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=11, color=BLUE)
ax.axhline(acc[0], ls=":", color=RED, lw=1.2)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8.8)
ax.set_ylabel("Reconciler truth-recovery on 13 E3 traps (%)")
ax.set_ylim(0, 112)
ax.set_title("C5-5  Reconciler information-bottleneck — minimal sufficient "
             "input")
ax.text(0.5, -0.34,
        "Accuracy SATURATES at L0 (answers only; all reasoning chains "
        "withheld). The reconciler's power is independent HINT-FREE "
        "RE-DERIVATION triggered by disagreement — not error-diagnosis of "
        f"chains (chain content inert here). Honest, n={d['n']}, "
        f"dq_ok={d['data_quality_ok']}. Tightens C4-4/C4-5: dissenter = "
        "pure TRIGGER; the conflict signal stays necessary (cannot tell "
        "misled from clean a priori).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
        color=GREY, wrap=True)
fig.subplots_adjust(bottom=0.34)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG/"c5_5_info_bottleneck.png", dpi=180, bbox_inches="tight")
print(f"saved {FIG/'c5_5_info_bottleneck.png'}")
