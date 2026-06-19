#!/usr/bin/env python3
"""Render a branch-and-merge tree diagram from a JSONL log.

For a problem where branching triggered (Feature F), shows:
  - Root problem
  - Fork into N branches with their diversity hints
  - Each branch's iteration history and code sketch
  - Evaluator scores
  - Winning branch highlighted, merged result on the right

Requires the branch_merge.py logging fix (BRANCH_CREATED + BRANCH_MERGED events).

Usage:
  python experiments/viz/visualize_branch_merge.py \\
    --log-file experiments/humaneval/results/logs/HumanEval_91.jsonl \\
    --out experiments/viz/figures/branch_merge_91.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_events(log_path: Path):
    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def find_branch_events(events):
    branches = []  # list of {fork_id, branch_index, diversity_hint, n_branches}
    merge = None   # {winning_fork, all_results, passed}
    for evt in events:
        etype = evt.get("event_type") or evt.get("event") or ""
        details = evt.get("details", {}) or {}
        if etype == "branch_created":
            branches.append({
                "fork_id": details.get("fork_id"),
                "branch_index": details.get("branch_index"),
                "diversity_hint": details.get("diversity_hint", ""),
                "n_branches": details.get("n_branches"),
            })
        elif etype == "branch_merged":
            merge = {
                "winning_fork": details.get("winning_fork"),
                "all_results": details.get("all_results", []),
                "passed": details.get("passed"),
            }
    return branches, merge


def draw_box(ax, x, y, w, h, text, color, edge="black", fontsize=9, fontweight="normal"):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="black", width=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=width),
    )


def render(log_path: Path, out_path: Path):
    events = parse_events(log_path)
    branches, merge = find_branch_events(events)

    if not branches:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                f"No BRANCH_CREATED events found in {log_path.name}.\n"
                f"This problem did not trigger Feature F (branch-and-merge).\n\n"
                f"Try a different log file, or check that the branch_merge.py\n"
                f"logging fix is in place for new runs.",
                ha="center", va="center", fontsize=11, wrap=True)
        ax.set_axis_off()
        ax.set_title(f"Branch-and-Merge — {log_path.stem}",
                     fontsize=12, fontweight="bold")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved (placeholder): {out_path}")
        return

    n_branches = len(branches)
    fig, ax = plt.subplots(figsize=(max(10, 3 * n_branches), 8))
    ax.set_xlim(0, 10 + 3 * (n_branches - 1))
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    # Root
    draw_box(ax, 0.3, 7.5, 3, 1.5, f"Root problem\n{log_path.stem}",
             "#bbdefb", fontweight="bold")

    # Branches in middle
    branch_y = 4.5
    branch_x_start = 4.5
    branch_w = 2.4
    branch_h = 2.5
    for i, b in enumerate(branches):
        bx = branch_x_start + i * (branch_w + 0.4)
        winning = (merge and merge.get("winning_fork") == b["fork_id"])
        color = "#c8e6c9" if winning else "#eeeeee"
        edge = "green" if winning else "black"
        text = f"Branch {b['branch_index']}\nhint: {b['diversity_hint'][:30]}"
        if winning:
            text += "\n★ WINNER"
        draw_box(ax, bx, branch_y, branch_w, branch_h, text,
                 color, edge=edge, fontweight="bold" if winning else "normal")
        # Arrow from root
        draw_arrow(ax, 1.8, 7.5, bx + branch_w / 2, branch_y + branch_h)

    # Merge result on the right
    if merge:
        mx = branch_x_start + n_branches * (branch_w + 0.4) + 0.5
        passed = merge.get("passed")
        color = "#c8e6c9" if passed else "#ffcdd2"
        result_text = f"Merged\n{'PASS' if passed else 'FAIL'}"
        draw_box(ax, mx, 4.8, 2.5, 2.0, result_text, color, fontweight="bold")
        for i in range(n_branches):
            bx = branch_x_start + i * (branch_w + 0.4)
            draw_arrow(ax, bx + branch_w, branch_y + branch_h / 2,
                       mx, 5.8)

    fig.suptitle(f"Branch-and-Merge Tree — {log_path.stem}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    render(log_path, Path(args.out))


if __name__ == "__main__":
    main()
