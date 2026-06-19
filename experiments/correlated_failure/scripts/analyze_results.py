#!/usr/bin/env python3
"""Analyze E3 (Correlated Failure) results and produce figures.

Generates:
  1. accuracy_bar.png      — accuracy per condition (Single | Majority | GAIA)
  2. per_problem_heatmap.png — 15 problems x 3 conditions, green/red, annotated
  3. trap_category_breakdown.png — accuracy split by trap category
  4. summary.txt            — text summary of key statistics

Usage:
  python experiments/correlated_failure/scripts/analyze_results.py
  python experiments/correlated_failure/scripts/analyze_results.py --results <path>
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"


CONDITION_LABELS = {
    "single": "Single Solver",
    "majority_vote": "Majority Vote\n(2 misled + 1 clean)",
    "gaia": "GAIA\n(+ reconciler)",
}

CONDITION_COLORS = {
    "single": "#cccccc",
    "majority_vote": "#f4a261",
    "gaia": "#2a9d8f",
}


def find_latest_results():
    candidates = sorted(RESULTS_DIR.glob("correlated_failure_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_accuracy_bar(data, out_path):
    conditions = [c for c in ["single", "majority_vote", "gaia"] if c in data]
    accuracies = [data[c]["summary"]["accuracy"] for c in conditions]
    labels = [CONDITION_LABELS[c] for c in conditions]
    colors = [CONDITION_COLORS[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "E3: Correlated Failure — GAIA's Reconciler Overrides Wrong Majority",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_problem_heatmap(data, out_path):
    conditions = [c for c in ["single", "majority_vote", "gaia"] if c in data]
    if not conditions:
        return

    # Get all unique problem IDs sorted
    problem_ids = sorted({
        r["problem_id"]
        for c in conditions for r in data[c]["results"]
    })

    matrix = np.zeros((len(problem_ids), len(conditions)))
    for j, c in enumerate(conditions):
        result_by_pid = {r["problem_id"]: r for r in data[c]["results"]}
        for i, pid in enumerate(problem_ids):
            r = result_by_pid.get(pid)
            matrix[i, j] = 1 if (r and r.get("passed")) else 0

    fig, ax = plt.subplots(figsize=(7, max(5, len(problem_ids) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS[c].replace("\n", " ") for c in conditions],
                       rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(problem_ids)))
    ax.set_yticklabels(problem_ids, fontsize=8)

    # Annotate with answers where available
    for j, c in enumerate(conditions):
        result_by_pid = {r["problem_id"]: r for r in data[c]["results"]}
        for i, pid in enumerate(problem_ids):
            r = result_by_pid.get(pid)
            if r is None:
                continue
            ans = r.get("proposed_answer")
            txt = "✓" if r.get("passed") else f"{ans}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")

    ax.set_title("E3: Per-problem accuracy (✓ = correct; number = wrong answer)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_trap_category_breakdown(data, out_path):
    conditions = [c for c in ["single", "majority_vote", "gaia"] if c in data]
    if not conditions:
        return

    # accuracy[condition][category] = accuracy
    accuracy = defaultdict(lambda: defaultdict(list))
    for c in conditions:
        for r in data[c]["results"]:
            cat = r.get("trap_category") or "other"
            accuracy[c][cat].append(1 if r.get("passed") else 0)

    categories = sorted({cat for c in conditions for cat in accuracy[c]})

    width = 0.8 / len(conditions)
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, c in enumerate(conditions):
        vals = [
            sum(accuracy[c][cat]) / len(accuracy[c][cat]) if accuracy[c][cat] else 0
            for cat in categories
        ]
        ax.bar(
            x + i * width - width * (len(conditions) - 1) / 2,
            vals, width,
            label=CONDITION_LABELS[c].replace("\n", " "),
            color=CONDITION_COLORS[c], edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("E3: Accuracy by Trap Category", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def write_summary(data, out_path):
    lines = ["=" * 80,
             "E3: Correlated Failure — Summary",
             "=" * 80, ""]

    for cond in ["single", "majority_vote", "gaia"]:
        if cond not in data:
            continue
        s = data[cond]["summary"]
        lines.append(f"--- {CONDITION_LABELS[cond].replace(chr(10), ' ')} ---")
        lines.append(f"  problems   : {s['n_problems']}")
        lines.append(f"  passed     : {s['n_passed']}")
        lines.append(f"  accuracy   : {s['accuracy']:.1%}")
        lines.append(f"  cost       : ${s['total_cost_usd']:.4f}")
        lines.append("")

    # Key result: how often did the reconciler save GAIA from a wrong majority?
    if "gaia" in data:
        gaia_results = data["gaia"]["results"]
        saved_by_reconciler = sum(
            1 for r in gaia_results
            if r.get("reconciler_sided_with_clean")
        )
        corr_fail = sum(
            1 for r in gaia_results if r.get("correlated_failure_present")
        )
        conflicts = sum(
            1 for r in gaia_results if r.get("conflict_detected")
        )
        lines.append("--- Key mechanism stats (GAIA) ---")
        lines.append(f"  correlated failure present (2 misled agreed wrong): "
                     f"{corr_fail} / {len(gaia_results)}")
        lines.append(f"  conflict detected by aggregator: "
                     f"{conflicts} / {len(gaia_results)}")
        lines.append(f"  reconciler sided with clean dissenter: "
                     f"{saved_by_reconciler} / {len(gaia_results)}")

    # Comparison: GAIA vs majority_vote
    if "gaia" in data and "majority_vote" in data:
        gaia_acc = data["gaia"]["summary"]["accuracy"]
        mv_acc = data["majority_vote"]["summary"]["accuracy"]
        gain = gaia_acc - mv_acc
        lines.append("")
        lines.append(f"GAIA vs Majority Vote: +{gain:.1%} accuracy gain")

    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")
    print()
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to results JSON (default: latest)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results) if args.results else find_latest_results()
    if not results_path or not results_path.exists():
        print(f"No results found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"Loading: {results_path}")
    data = load_results(results_path)

    print("\nGenerating figures...")
    plot_accuracy_bar(data, FIGURES_DIR / "accuracy_bar.png")
    plot_per_problem_heatmap(data, FIGURES_DIR / "per_problem_heatmap.png")
    plot_trap_category_breakdown(data, FIGURES_DIR / "trap_category_breakdown.png")
    write_summary(data, FIGURES_DIR / "summary.txt")


if __name__ == "__main__":
    main()
