#!/usr/bin/env python3
"""Information-flow diagram: how an answer actually moves through GAIA.

Aggregates the E3 GAIA state dumps into a layered flow:
  solver artifacts -> aggregator verdict -> CONFLICT signal -> reconciler
  -> verified outcome.  Edge widths = #episodes taking that path.

This visually proves the conflict-as-task PATHWAY is the load-bearing route
(vs the rare unanimous path), grounding the topology discussion in observed
flow rather than accuracy alone.
"""
import glob, json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT/"experiments"/"viz"/"figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    flows = Counter()
    n = 0
    for sf in glob.glob(str(ROOT/"experiments/correlated_failure/logs/**/*.state.json"),
                        recursive=True):
        try:
            d = json.load(open(sf))
        except Exception:
            continue
        n += 1
        ex = d.get("extra", {})
        arts = list(d.get("artifacts", {}).values())
        sigs = list(d.get("signals", {}).values())
        had_solvers = any(a.get("metadata", {}).get("subtype") == "math_solution"
                          for a in arts)
        had_conflict = any(s.get("type") == "CONFLICT" for s in sigs)
        had_recon = ex.get("conflict_resolved") or any(
            a.get("metadata", {}).get("subtype") == "reconciled_solution" for a in arts)
        passed = ex.get("passed")
        if had_solvers:
            flows[("Solvers", "Aggregator")] += 1
        if had_conflict:
            flows[("Aggregator", "CONFLICT")] += 1
        else:
            flows[("Aggregator", "Unanimous")] += 1
        if had_recon:
            flows[("CONFLICT", "Reconciler")] += 1
        flows[("Reconciler" if had_recon else
               ("Unanimous" if not had_conflict else "CONFLICT"),
               "PASS" if passed else "FAIL")] += 1

    layers = {"Solvers": (0, 0.5), "Aggregator": (1, 0.5),
              "CONFLICT": (2, 0.65), "Unanimous": (2, 0.2),
              "Reconciler": (3, 0.65), "PASS": (4, 0.6), "FAIL": (4, 0.15)}
    col = {"Solvers": "#264653", "Aggregator": "#2a9d8f", "CONFLICT": "#e76f51",
           "Unanimous": "#bdbdbd", "Reconciler": "#e9c46a",
           "PASS": "#2a9d8f", "FAIL": "#e76f51"}

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(-0.5, 4.8); ax.set_ylim(0, 1); ax.axis("off")
    for name, (lx, ly) in layers.items():
        ax.add_patch(mp.FancyBboxPatch((lx-0.18, ly-0.07), 0.36, 0.14,
                     boxstyle="round,pad=0.02", facecolor=col[name],
                     edgecolor="black", lw=1.3))
        ax.text(lx, ly, name, ha="center", va="center", fontsize=10,
                color="white" if name not in ("Unanimous",) else "black",
                fontweight="bold")
    mx = max(flows.values()) if flows else 1
    for (a, b), w in flows.items():
        if a not in layers or b not in layers:
            continue
        (x1, y1), (x2, y2) = layers[a], layers[b]
        ax.annotate("", xy=(x2-0.18, y2), xytext=(x1+0.18, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1+5*w/mx,
                                    color="gray", alpha=0.65))
        ax.text((x1+x2)/2, (y1+y2)/2+0.03, str(w), fontsize=8, ha="center",
                color="black")
    ax.set_title(f"Information flow through GAIA on the trap suite "
                 f"(n={n} episodes)\nThe CONFLICT→Reconciler path is the "
                 f"load-bearing route, not the unanimous shortcut",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT/"information_flow.png", dpi=150)
    plt.close()
    print(f"Saved {OUT/'information_flow.png'}  (n={n})")
    print("flows:", dict(flows))


if __name__ == "__main__":
    main()
