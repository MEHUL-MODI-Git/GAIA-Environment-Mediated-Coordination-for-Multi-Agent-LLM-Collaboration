#!/usr/bin/env python3
"""Analyze E4 (Coverage Scaling) results and produce figures.

Generates:
  1. accuracy_vs_coverage.png  — line chart, GAIA solid + Isolated dashed
  2. accuracy_drop_summary.png — bar showing how steeply isolated drops vs gaia
  3. summary.txt               — text summary
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results" / "coverage"
FIGURES_DIR = EXPERIMENT_DIR / "figures" / "coverage"


def find_latest_results():
    candidates = sorted(RESULTS_DIR.glob("coverage_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def plot_accuracy_vs_coverage(data, out_path):
    by_cond = defaultdict(list)
    for key, val in data.items():
        s = val["summary"]
        by_cond[s["condition"]].append((s["coverage"], s["accuracy"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "gaia": {"color": "#2a9d8f", "linestyle": "-", "marker": "o",
                  "linewidth": 2.5, "label": "GAIA (blackboard)"},
        "isolated": {"color": "#e76f51", "linestyle": "--", "marker": "s",
                      "linewidth": 2.5, "label": "Isolated (no sharing)"},
    }

    for cond, pts in by_cond.items():
        pts.sort()
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] for p in pts]
        style = styles.get(cond, {"label": cond})
        ax.plot(xs, ys, **style)

    ax.set_xlabel("Per-agent Clue Coverage (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("E4: Graceful Degradation under Information Asymmetry",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def write_summary(data, out_path):
    lines = ["=" * 80, "E4: Information Asymmetry Scaling — Summary", "=" * 80, ""]

    rows = []
    for key, val in sorted(data.items()):
        s = val["summary"]
        rows.append((s["condition"], s["coverage"], s["accuracy"], s["total_cost_usd"]))

    rows.sort(key=lambda r: (r[0], r[1]))

    lines.append(f"{'Condition':<12} {'Coverage':<10} {'Accuracy':<10} {'Cost'}")
    lines.append("─" * 50)
    for cond, cov, acc, cost in rows:
        lines.append(f"{cond:<12} {cov:.0%}      {acc:.1%}      ${cost:.4f}")

    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")
    print()
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results) if args.results else find_latest_results()
    if not results_path or not results_path.exists():
        print(f"No results found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Loading: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    print("\nGenerating figures...")
    plot_accuracy_vs_coverage(data, FIGURES_DIR / "accuracy_vs_coverage.png")
    write_summary(data, FIGURES_DIR / "summary.txt")


if __name__ == "__main__":
    main()
