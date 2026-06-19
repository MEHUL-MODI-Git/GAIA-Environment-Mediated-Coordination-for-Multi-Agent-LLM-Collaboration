#!/usr/bin/env python3
"""C5-5b figure — the reconciler's input-independence is ROBUST.

Grouped bars (orig n=13 vs expanded n=32) across 5 conditions. Flat at
~97-100% everywhere — adversarial majority-prior framing and hidden-dissenter
do NOT move it. The reconciler's only causal dependency is being TRIGGERED;
all inputs are inert. Honest: expanded steady 97% = one hard trap missed in
EVERY condition (capability ceiling, not a framing effect).
"""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).parent.parent/"results"
FIG = Path(__file__).parent.parent/"figures"
BLUE, TEAL, GREY = "#1565C0", "#00695C", "#455A64"

d = json.load(open(sorted(glob.glob(str(RES/"c5_5b_*.json")))[-1]))
conds = ["neutral_L0", "neutral_L2", "adversarial_L0", "adversarial_L2",
         "dissenter_hidden_L0"]
lab = ["neutral\nL0", "neutral\nL2", "adversarial\nL0", "adversarial\nL2",
       "dissenter\nhidden L0"]
fig, ax = plt.subplots(figsize=(9.2, 5.0))
w = 0.38
for (sset, off, col) in [("orig", -w/2, TEAL), ("expanded", w/2, BLUE)]:
    C = d["by_set"][sset]["conditions"]
    m = [C[k]["acc_ci"][0]*100 for k in conds]
    lo = [(C[k]["acc_ci"][0]-C[k]["acc_ci"][1])*100 for k in conds]
    hi = [(C[k]["acc_ci"][2]-C[k]["acc_ci"][0])*100 for k in conds]
    x = np.arange(5)+off
    b = ax.bar(x, m, w, yerr=[lo, hi], capsize=3, color=col,
               edgecolor="k", lw=0.6,
               label=f"{sset} (n={d['by_set'][sset]['n']})")
    ax.bar_label(b, labels=[f"{v:.0f}" for v in m], padding=2, fontsize=8.5)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(lab, fontsize=9)
ax.set_ylabel("Reconciler truth-recovery (%), bootstrap 95% CI")
ax.set_ylim(0, 113)
ax.set_title("C5-5b  The reconciler's input-independence is ROBUST")
ax.legend(fontsize=9, loc="lower center")
ax.text(0.5, -0.26,
        "Adversarial framing (\"the majority is ~90% reliable, treat as a "
        "strong prior\") and dissenter-hidden (only the WRONG majority "
        "answer shown, no chains) leave accuracy UNCHANGED across both "
        "substrates. The reconciler's sole causal dependency is being "
        "TRIGGERED; every upstream input is inert. (Expanded steady 97% = "
        "one hard trap missed in every condition — capability ceiling, not "
        "framing.)",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
        color=GREY, wrap=True)
fig.subplots_adjust(bottom=0.30)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG/"c5_5b_robust.png", dpi=180, bbox_inches="tight")
print(f"saved {FIG/'c5_5b_robust.png'}")
