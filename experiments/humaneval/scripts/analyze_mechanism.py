#!/usr/bin/env python3
"""E5: Mechanism Analysis from existing HumanEval JSONL logs.

This experiment is FREE — no new agent runs. It mines the existing 164 HumanEval
JSONL logs to extract:

  1. Signal trigger rates: how often did CONFLICT / UNCERTAINTY / DUPLICATION
     signals fire across the run? (Bar chart.)

  2. Conflict resolution breakdown: when a conflict fired, was it resolved on
     the first fix-iteration, later, or never? (Stacked bar.)

  3. Iterations-to-solution distribution: histogram across all problems.

  4. Branch usage (Feature F) — count of BRANCH_CREATED + BRANCH_MERGED events,
     broken down by whether the winning branch passed.

Generates figures in:
  experiments/humaneval/figures/mechanism/

Usage:
  python experiments/humaneval/scripts/analyze_mechanism.py
  python experiments/humaneval/scripts/analyze_mechanism.py --logs-dir <path>
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent.parent
DEFAULT_LOGS_DIR = EXPERIMENT_DIR / "results" / "logs"
FIGURES_DIR = EXPERIMENT_DIR / "figures" / "mechanism"


def parse_log_file(path: Path) -> dict:
    """Parse one JSONL log file and extract mechanism stats for one problem."""
    info = {
        "problem_id": None,
        "passed": None,
        "iterations": 0,
        "duration_s": 0,
        "signals": Counter(),
        "signal_resolved_at_iteration": [],
        "branch_created": 0,
        "branch_merged": 0,
        "branch_merge_passed": None,
        "n_llm_calls": 0,
        "total_cost_usd": 0.0,
    }
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("event_type") or evt.get("event") or ""
                details = evt.get("details", {}) or {}
                if etype == "episode_start":
                    info["problem_id"] = details.get("problem_id") or evt.get("task_id")
                elif etype == "episode_end":
                    info["passed"] = details.get("passed") or evt.get("passed")
                    metrics = evt.get("metrics") or {}
                    info["duration_s"] = (
                        metrics.get("duration_ms", 0) / 1000.0
                        or evt.get("elapsed_s", 0)
                    )
                elif etype == "iteration_start":
                    info["iterations"] = max(info["iterations"], details.get("iteration", 0))
                elif etype == "signal_posted":
                    sig_type = details.get("type") or details.get("signal_type", "UNKNOWN")
                    info["signals"][sig_type] += 1
                elif etype == "branch_created":
                    info["branch_created"] += 1
                elif etype == "branch_merged":
                    info["branch_merged"] += 1
                    info["branch_merge_passed"] = details.get("passed")
                elif etype == "llm_call":
                    info["n_llm_calls"] += 1
                    info["total_cost_usd"] += details.get("cost_usd", 0)
    except Exception as e:
        print(f"Warning: failed to parse {path}: {e}")
    return info


def collect_all_logs(logs_dir: Path) -> list:
    files = sorted(logs_dir.glob("*.jsonl"))
    print(f"Found {len(files)} JSONL log files in {logs_dir}")
    return [parse_log_file(f) for f in files]


def plot_signal_rates(records, out_path):
    total = len(records)
    signal_totals = Counter()
    for r in records:
        for sig, count in r["signals"].items():
            if count > 0:
                signal_totals[sig] += 1  # count problems where this signal appeared

    types = ["CONFLICT", "UNCERTAINTY", "DUPLICATION", "URGENCY", "STALENESS", "BUDGET_RISK"]
    types_present = [t for t in types if signal_totals.get(t, 0) > 0]
    types_present += [t for t in signal_totals if t not in types]
    if not types_present:
        print(f"  (no signals found in logs — skipping {out_path.name})")
        return

    counts = [signal_totals[t] for t in types_present]
    percents = [100 * c / total if total else 0 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types_present, percents,
                   color="#2a9d8f", edgecolor="black", linewidth=1.2)
    for bar, c, p in zip(bars, counts, percents):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"% of problems triggering signal (n={total})")
    ax.set_title("E5: Signal Trigger Rates Across 164 HumanEval Problems",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_iteration_histogram(records, out_path):
    iterations = [r["iterations"] for r in records if r["iterations"] > 0]
    passed_iters = [r["iterations"] for r in records if r["iterations"] > 0 and r["passed"]]
    failed_iters = [r["iterations"] for r in records if r["iterations"] > 0 and not r["passed"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = range(1, max(iterations + [10]) + 2)
    ax.hist([passed_iters, failed_iters], bins=bins,
            stacked=True,
            label=["Passed", "Failed"],
            color=["#2a9d8f", "#e76f51"],
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Iterations to terminate")
    ax.set_ylabel("Number of problems")
    ax.set_title("E5: Iterations-to-Solution Distribution",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_branch_usage(records, out_path):
    branched = [r for r in records if r["branch_created"] > 0]
    if not branched:
        # Branch logging was added at the end of this work — older logs won't have it
        print(f"  (no branch events found — skipping {out_path.name})")
        # Write a placeholder note
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5,
                "No BRANCH_CREATED events found in these logs.\n"
                "(Branch logging was added after this run.)\n"
                "Re-run on logs generated after the branch_merge.py fix to populate.",
                ha="center", va="center", fontsize=11, wrap=True)
        ax.set_axis_off()
        ax.set_title("E5: Branch-and-Merge Usage", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    passed = sum(1 for r in branched if r.get("branch_merge_passed"))
    failed = len(branched) - passed
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Branched + Passed", "Branched + Failed"],
           [passed, failed],
           color=["#2a9d8f", "#e76f51"], edgecolor="black", linewidth=1.2)
    for i, v in enumerate([passed, failed]):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom",
                fontweight="bold", fontsize=12)
    ax.set_ylabel("Number of problems")
    ax.set_title(f"E5: Branch-and-Merge Outcomes ({len(branched)} of {len(records)} problems)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def write_summary(records, out_path):
    lines = ["=" * 80, "E5: Mechanism Analysis — HumanEval Logs", "=" * 80, ""]
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    failed = total - passed

    lines.append(f"Total problems analyzed: {total}")
    lines.append(f"Passed                 : {passed} ({passed/total if total else 0:.1%})")
    lines.append(f"Failed                 : {failed}")
    lines.append("")

    # Signal counts
    signal_totals = Counter()
    for r in records:
        for sig, count in r["signals"].items():
            signal_totals[sig] += count

    lines.append("--- Signal events (total occurrences) ---")
    for sig, count in signal_totals.most_common():
        n_problems = sum(1 for r in records if r["signals"].get(sig, 0) > 0)
        lines.append(f"  {sig:<14}: {count:4d} total, in {n_problems} problems "
                     f"({n_problems/total if total else 0:.1%})")
    lines.append("")

    # Iteration stats
    iters = [r["iterations"] for r in records if r["iterations"] > 0]
    if iters:
        lines.append("--- Iteration stats ---")
        lines.append(f"  mean iterations: {np.mean(iters):.1f}")
        lines.append(f"  median         : {int(np.median(iters))}")
        lines.append(f"  min            : {min(iters)}")
        lines.append(f"  max            : {max(iters)}")
        lines.append("")

    # Branch stats
    branched = [r for r in records if r["branch_created"] > 0]
    lines.append(f"--- Branch-and-Merge (Feature F) ---")
    lines.append(f"  Problems where branching triggered: {len(branched)}")
    if branched:
        branch_passed = sum(1 for r in branched if r.get("branch_merge_passed"))
        lines.append(f"  Of those, branch winner passed: {branch_passed}/{len(branched)}")

    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path}")
    print()
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=str, default=str(DEFAULT_LOGS_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)

    if not logs_dir.exists():
        print(f"Logs dir not found: {logs_dir}")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records = collect_all_logs(logs_dir)
    if not records:
        print("No records found")
        sys.exit(1)

    print("\nGenerating mechanism analysis figures...")
    plot_signal_rates(records, FIGURES_DIR / "signal_rates.png")
    plot_iteration_histogram(records, FIGURES_DIR / "iteration_histogram.png")
    plot_branch_usage(records, FIGURES_DIR / "branch_usage.png")
    write_summary(records, FIGURES_DIR / "summary.txt")


if __name__ == "__main__":
    main()
