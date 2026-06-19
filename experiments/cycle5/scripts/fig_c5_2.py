#!/usr/bin/env python3
"""C5-2 figure — exact Shapley + coalition lattice (reconciler PROTECTS)."""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent.parent/"results"
FIG = Path(__file__).parent.parent/"figures"
BLUE, GREEN, RED, GREY, PURPLE = "#1565C0", "#2E7D32", "#B71C1C", "#455A64", "#6A1B9A"

d = json.load(open(sorted(glob.glob(str(RES/"c5_2_*.json")))[-1]))
sh = d["shapley"]; co = d["coalitions"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))

roles = ["misled0", "misled1", "clean", "reconciler"]
vals = [sh[r] for r in roles]
cols = [GREY, GREY, GREEN, BLUE]
b = ax1.bar(range(4), vals, color=cols, edgecolor="k", lw=0.6)
ax1.bar_label(b, labels=[f"{v:+.3f}" for v in vals], padding=4, fontsize=10)
ax1.axhline(0, color="k", lw=0.8)
ax1.set_xticks(range(4))
ax1.set_xticklabels(["misled0\n(redundant)", "misled1\n(redundant)",
                     "clean\n(dissenter)", "reconciler"], fontsize=9)
ax1.set_ylabel("Exact Shapley value φ (credit on the 13-trap game)")
ax1.set_ylim(-0.1, 0.8)
ax1.set_title("C5-2  Exact Shapley attribution (all 16 coalitions)")
ea = d["efficiency_axiom"]
ax1.text(0.0, -0.32,
         f"Efficiency axiom Σφ={ea['sum_phi']}=v(N)−v(∅)="
         f"{ea['v_grand_minus_empty']} ✓ (exact).\nRedundant misled solvers "
         f"≈0 (slightly negative) credit;\nclean+reconciler carry it all.\n"
         f"Interaction I(clean,reconciler)=+"
         f"{d['interaction_clean_reconciler']}\n— strong complementarity, "
         f"not substitutes.",
         transform=ax1.transAxes, ha="left", va="top",
         fontsize=8.2, color=GREY)

key = [("clean", "clean\nALONE"),
       ("clean,misled0", "clean+misled\n(NO reconciler)"),
       ("clean,misled0,misled1", "clean+2 misled\n(NO reconciler)"),
       ("clean,reconciler", "clean+reconciler"),
       ("clean,misled0,misled1,reconciler", "full GAIA\npipeline")]
kv = [co[k]*100 for k, _ in key]
kc = [GREEN, RED, RED, BLUE, BLUE]
b2 = ax2.bar(range(5), kv, color=kc, edgecolor="k", lw=0.6)
ax2.bar_label(b2, labels=[f"{v:.0f}%" for v in kv], padding=4, fontsize=10)
ax2.set_xticks(range(5))
ax2.set_xticklabels([l for _, l in key], fontsize=8.2)
ax2.set_ylabel("Coalition accuracy v(S) on 13 traps (%)")
ax2.set_ylim(0, 112)
ax2.set_title("C5-2  Coalition lattice — the reconciler PROTECTS a "
              "pre-existing signal")
ax2.text(1.0, -0.32,
         "Hint-free clean solver ALONE = 100%.\nMajority-pooling it with "
         "misled peers\n(no reconciler) DESTROYS it (100→15%).\nRe-adding the "
         "reconciler RESTORES it (→100%).\nThe reconciler shields a "
         "pre-existing\ncorrect signal from correlated-majority\nerasure "
         "(explains C5-5's 100%-at-answers-only).",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=8.2, color=GREY)

fig.suptitle("GAIA credit assignment: clean dissenter carries the signal, "
             "the reconciler PROTECTS it from correlated-majority erasure "
             "(gpt-4.1-nano, exact Shapley)", fontsize=11.5, y=1.02)
fig.subplots_adjust(bottom=0.34, wspace=0.22)
FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG/"c5_2_shapley_lattice.png", dpi=180, bbox_inches="tight")
print(f"saved {FIG/'c5_2_shapley_lattice.png'}")
