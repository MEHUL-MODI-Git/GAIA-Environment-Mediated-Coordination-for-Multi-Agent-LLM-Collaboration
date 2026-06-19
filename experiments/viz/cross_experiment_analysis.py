#!/usr/bin/env python3
"""NX4 + stats hardening: unified cross-experiment analysis.

Produces the paper's two highest-leverage *free* artifacts (no new runs):

  1. cost_accuracy_pareto.png — every condition from E3/E4/E8/E9 plotted as
     accuracy vs $/task, with the Pareto frontier drawn. One figure that
     summarizes the entire efficiency argument.

  2. headline_stats.md / .json — every headline cell with a bootstrap 95% CI
     (10k resamples of the per-problem pass vector). Fixes the "single seed,
     small n, no CI" credibility gap (roadmap G7).

Honest scope note baked into the output: these are CONSTRUCTED diagnostic
suites (hand-built traps/puzzles), not random samples — CIs quantify
within-suite precision, not population generalization.
"""

import json
import glob
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).parent.parent.parent
OUT = PROJECT / "experiments" / "viz" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def latest(pat):
    fs = sorted(glob.glob(str(PROJECT / pat)))
    return fs[-1] if fs else None


def bootstrap_ci(passed_bools, n_boot=10000, seed=0):
    """95% bootstrap CI for the mean of a 0/1 vector."""
    if not passed_bools:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(passed_bools)
    means = []
    for _ in range(n_boot):
        s = sum(passed_bools[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    point = sum(passed_bools) / n
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (point, lo, hi)


def collect():
    """Return list of {exp, condition, label, acc, lo, hi, cost_per_task, n}."""
    rows = []

    def add(exp, cond, label, results):
        passed = [1 if r.get("passed") else 0 for r in results]
        costs = [r.get("cost_usd", 0) or 0 for r in results]
        if not passed:
            return
        pt, lo, hi = bootstrap_ci(passed)
        n = len(passed)
        cpt = (sum(costs) / n) if n else 0
        rows.append({"exp": exp, "condition": cond, "label": label,
                     "acc": pt, "lo": lo, "hi": hi,
                     "cost_per_task": cpt, "n": n})

    # E3
    f = latest("experiments/correlated_failure/results/correlated_failure_*.json")
    if f:
        d = json.load(open(f))
        for c in ("single", "majority_vote", "gaia"):
            if c in d:
                add("E3", c, f"E3:{c}", d[c]["results"])

    # E4 (coverage)
    f = latest("experiments/puzzle/results/coverage/coverage_*.json")
    if f:
        d = json.load(open(f))
        for k, v in d.items():
            s = v["summary"]
            add("E4", k, f"E4:{s['condition']}@{int(s['coverage']*100)}%", v["results"])

    # E8 (scaling)
    f = latest("experiments/puzzle/results/scaling/scaling_*.json")
    if f:
        d = json.load(open(f))
        for k, v in d.items():
            s = v["summary"]
            add("E8", k, f"E8:{s['condition_type']}-n{s['num_agents']}", v["results"])

    # E9 (merged)
    f = str(PROJECT / "experiments/fault_injection/results/fault_injection_MERGED.json")
    if Path(f).exists():
        d = json.load(open(f))
        for c, v in d.items():
            add("E9", c, f"E9:{c}", v["results"])

    return rows


def pareto_front(points):
    """points: list of (cost, acc, label). Return indices on the frontier
    (minimize cost, maximize acc)."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best_acc = [], -1
    for i in idx:
        if points[i][1] > best_acc:
            front.append(i)
            best_acc = points[i][1]
    return set(front)


def plot_pareto(rows, out):
    pts = [(r["cost_per_task"], r["acc"], r["label"]) for r in rows]
    front = pareto_front(pts)
    exp_color = {"E3": "#2a9d8f", "E4": "#e76f51", "E8": "#264653", "E9": "#e9c46a"}

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, r in enumerate(rows):
        on = i in front
        ax.scatter(r["cost_per_task"], r["acc"],
                   s=140 if on else 70,
                   c=exp_color.get(r["exp"], "#888"),
                   edgecolors="black", linewidths=1.6 if on else 0.6,
                   marker="*" if "gaia" in r["condition"] else "o",
                   zorder=3 if on else 2, alpha=0.9)
        # error bar (bootstrap CI)
        ax.plot([r["cost_per_task"]] * 2, [r["lo"], r["hi"]],
                color=exp_color.get(r["exp"], "#888"), alpha=0.35, lw=1.2, zorder=1)

    # frontier line
    fr = sorted([pts[i] for i in front])
    ax.plot([p[0] for p in fr], [p[1] for p in fr], "k--", alpha=0.5,
            label="Pareto frontier", zorder=1)
    for i in front:
        ax.annotate(rows[i]["label"], (rows[i]["cost_per_task"], rows[i]["acc"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=10, label=e)
               for e, c in exp_color.items()]
    handles.append(plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='gray', markersize=14, label='GAIA variant'))
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xlabel("Cost per task (USD)", fontsize=12)
    ax.set_ylabel("Accuracy (bootstrap 95% CI bars)", fontsize=12)
    ax.set_title("NX4: Cost–Accuracy Pareto Frontier across all GAIA experiments",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
    return front


def main():
    rows = collect()
    front = plot_pareto(rows, OUT / "cost_accuracy_pareto.png")

    lines = ["# NX4 + Stats — Headline results with bootstrap 95% CIs", "",
             "Scope: CONSTRUCTED diagnostic suites (hand-built traps/puzzles), "
             "not random population samples. CIs = within-suite precision "
             "(10k bootstrap resamples of the per-problem pass vector).", "",
             "| Exp | Condition | n | Accuracy [95% CI] | $/task | On Pareto front |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(rows, key=lambda x: (x["exp"], -x["acc"]))):
        star = "★" if any(rows[j] is r for j in front) else ""
        lines.append(
            f"| {r['exp']} | {r['condition']} | {r['n']} | "
            f"{r['acc']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}] | "
            f"${r['cost_per_task']:.4f} | {star} |")
    (OUT / "headline_stats.md").write_text("\n".join(lines))
    json.dump(rows, open(OUT / "headline_stats.json", "w"), indent=2)
    print(f"  Saved: {OUT/'headline_stats.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
