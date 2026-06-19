#!/usr/bin/env python3
"""C5-1 + C5-1b centerpiece figure — the honest token-matched story.

LEFT  (C5-1): at EQUAL reasoning tokens, unstructured scaling fails
              (self-cons / extended 23%); structured correction succeeds
              (GAIA & self-refine 100%). Thesis = structure, not multiplicity.
RIGHT (C5-1b): the decisive isolation. Unbiased self-refine ≈ GAIA, but when
              the lone agent's self-critique SHARES the bias it collapses
              (69-77%) while GAIA's INDEPENDENT reconciler holds (97-100%);
              fully-correlated debate is worst. Bootstrap 95% CIs; expanded
              set CIs are non-overlapping. Independence of the corrector =
              GAIA's distinct, honestly-bounded value.
"""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).parent.parent/"results"
FIG = Path(__file__).parent.parent/"figures"
BLUE, GREEN, RED, ORANGE, GREY = "#1565C0", "#2E7D32", "#B71C1C", "#E65100", "#455A64"

d1 = json.load(open(sorted(glob.glob(str(RES/"c5_1_2*.json")))[-1]))
d1b = json.load(open(sorted(glob.glob(str(RES/"c5_1b_*.json")))[-1]))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.2, 5.4))

# LEFT — C5-1 token-matched
B = d1["reasoning_token_budget_B"]
arms = [("GAIA-core", d1["gaia_core"]["accuracy"], BLUE),
        ("single-agent\nself-REFINE", d1["single_agent_arms"]["sa_refine"]["accuracy"], GREEN),
        ("single-agent\nself-consistency", d1["single_agent_arms"]["sa_selfcons"]["accuracy"], RED),
        ("single-agent\nextended pass", d1["single_agent_arms"]["sa_extended"]["accuracy"], RED)]
xs = range(4); ys = [a[1]*100 for a in arms]
b = axL.bar(xs, ys, color=[a[2] for a in arms], edgecolor="k", lw=0.6)
axL.bar_label(b, labels=[f"{y:.0f}%" for y in ys], padding=3, fontsize=11)
axL.set_xticks(list(xs)); axL.set_xticklabels([a[0] for a in arms], fontsize=8.8)
axL.set_ylabel("Accuracy on 13 traps (%)")
axL.set_ylim(0, 112)
axL.set_title(f"C5-1  Equal reasoning-token budget (B≈{B:.0f} tok)")
axL.text(0.0, -0.30,
         "Unstructured scaling (self-consistency, longer single pass)\n"
         "stays at 23% — correlated bias is NOT a token deficit.\n"
         "Structured self-correction (GAIA & self-refine) = 100%.\n"
         "→ the active ingredient is STRUCTURE, not agent count.",
         transform=axL.transAxes, ha="left", va="top", fontsize=8.2,
         color=GREY)

# RIGHT — C5-1b decisive independence (orig vs expanded, CI bars)
order = ["gaia_core", "sa_refine_unbiased", "sa_refine_biased",
         "debate_biased"]
lab = ["GAIA-core\n(independent\nreconciler)", "single self-refine\n(UNbiased\ncritique)",
       "single self-refine\n(BIASED\ncritique)", "debate\n(all biased)"]
cols = [BLUE, GREEN, ORANGE, RED]
w = 0.38
for gi, (sset, off) in enumerate([("orig", -w/2), ("expanded", w/2)]):
    A = d1b["by_set"][sset]["arms"]
    m = [A[k]["acc_ci"][0]*100 for k in order]
    lo = [(A[k]["acc_ci"][0]-A[k]["acc_ci"][1])*100 for k in order]
    hi = [(A[k]["acc_ci"][2]-A[k]["acc_ci"][0])*100 for k in order]
    x = np.arange(4)+off
    bb = axR.bar(x, m, w, yerr=[lo, hi], capsize=3,
                 color=cols, edgecolor="k", lw=0.6,
                 alpha=1.0 if sset == "expanded" else 0.55,
                 label=f"{sset} (n={d1b['by_set'][sset]['n']})")
axR.set_xticks(np.arange(4)); axR.set_xticklabels(lab, fontsize=8.2)
axR.set_ylabel("Accuracy (%), bootstrap 95% CI")
axR.set_ylim(0, 112)
axR.set_title("C5-1b  Independence of the corrector is decisive")
axR.legend(fontsize=8.5, loc="lower left")
axR.text(1.0, -0.30,
         "Unbiased self-refine ≈ GAIA (concession stands).\nWhen the lone "
         "agent's self-critique SHARES the bias\nit cannot self-correct "
         "(69-77%); GAIA's INDEPENDENT\nreconciler holds (97-100%; expanded "
         "CIs disjoint).\nFully-correlated debate is worst (47-54%).",
         transform=axR.transAxes, ha="right", va="top", fontsize=8.2,
         color=GREY)

fig.suptitle("GAIA's value is structured correction by an INDEPENDENT "
             "corrector — not agent count, not compute (gpt-4.1-nano, "
             "token-matched)", fontsize=11.5, y=1.03)
fig.subplots_adjust(bottom=0.30, wspace=0.22)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG/"c5_1_token_matched.png", dpi=180, bbox_inches="tight")
print(f"saved {FIG/'c5_1_token_matched.png'}")
