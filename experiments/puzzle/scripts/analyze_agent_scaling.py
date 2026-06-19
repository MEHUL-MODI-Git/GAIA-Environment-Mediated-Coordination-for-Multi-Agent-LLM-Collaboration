#!/usr/bin/env python3
"""Analyze E8 (Agent Scaling) results and produce figures.

Generates:
  1. accuracy_vs_count.png   — line chart, GAIA vs Homogeneous
  2. cost_vs_count.png       — cost per puzzle
  3. cost_per_correct.png    — efficiency: cost per correct answer
  4. summary.txt
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
RESULTS_DIR = EXPERIMENT_DIR / "results" / "scaling"
FIGURES_DIR = EXPERIMENT_DIR / "figures" / "scaling"


def find_latest_results():
    candidates = sorted(RESULTS_DIR.glob("scaling_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def collect_by_condition_type(data):
    """Return dict[condition_type] = list of (num_agents, accuracy, cost_per_puzzle)."""
    out = defaultdict(list)
    for key, val in data.items():
        s = val["summary"]
        per_puzzle = s["total_cost_usd"] / s["n_puzzles"] if s["n_puzzles"] else 0
        out[s["condition_type"]].append((s["num_agents"], s["accuracy"], per_puzzle))
    for v in out.values():
        v.sort()
    return out


def plot_accuracy_vs_count(by_type, out_path):
    styles = {
        "gaia": {"color": "#2a9d8f", "linestyle": "-", "marker": "o",
                  "linewidth": 2.5, "label": "GAIA (role-specialized)"},
        "homogeneous": {"color": "#e76f51", "linestyle": "--", "marker": "s",
                         "linewidth": 2.5, "label": "Homogeneous (no coordination)"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for ctype, rows in by_type.items():
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        style = styles.get(ctype, {"label": ctype})
        ax.plot(xs, ys, **style)

    ax.set_xlabel("Number of Agents", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("E8: Accuracy vs Agent Count — Role Diversity Matters",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cost_efficiency(by_type, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {
        "gaia": {"color": "#2a9d8f", "marker": "o", "label": "GAIA"},
        "homogeneous": {"color": "#e76f51", "marker": "s", "label": "Homogeneous"},
    }

    # Left: cost per puzzle
    for ctype, rows in by_type.items():
        xs = [r[0] for r in rows]
        costs = [r[2] for r in rows]
        s = styles.get(ctype, {"label": ctype})
        axes[0].plot(xs, costs, linewidth=2, **s)
    axes[0].set_xlabel("Number of Agents")
    axes[0].set_ylabel("Cost per Puzzle (USD)")
    axes[0].set_title("Cost per puzzle")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    # Right: cost per correct answer (efficiency)
    for ctype, rows in by_type.items():
        xs = [r[0] for r in rows]
        ratios = [r[2] / r[1] if r[1] > 0 else None for r in rows]
        s = styles.get(ctype, {"label": ctype})
        axes[1].plot(xs, ratios, linewidth=2, **s)
    axes[1].set_xlabel("Number of Agents")
    axes[1].set_ylabel("Cost / Accuracy (USD per fractional correct)")
    axes[1].set_title("Cost-to-correctness ratio (lower = better)")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.suptitle("E8: Cost-Quality Tradeoff", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def write_summary(by_type, out_path):
    lines = ["=" * 80, "E8: Agent Scaling — Summary", "=" * 80, ""]
    lines.append(f"{'Cond. Type':<14} {'N Agents':<10} {'Accuracy':<10} {'Cost/Puzzle':<12}")
    lines.append("─" * 50)
    for ctype, rows in by_type.items():
        for n_agents, acc, cost in rows:
            lines.append(f"{ctype:<14} {n_agents:<10} {acc:.1%}      ${cost:.4f}")
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

    by_type = collect_by_condition_type(data)

    print("\nGenerating figures...")
    plot_accuracy_vs_count(by_type, FIGURES_DIR / "accuracy_vs_count.png")
    plot_cost_efficiency(by_type, FIGURES_DIR / "cost_efficiency.png")
    write_summary(by_type, FIGURES_DIR / "summary.txt")


if __name__ == "__main__":
    main()
