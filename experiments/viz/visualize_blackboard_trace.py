#!/usr/bin/env python3
"""Render a blackboard state snapshot grid from a JSONL log.

Given a problem's log file, replays it event by event and snapshots the
blackboard state at user-specified iterations. Each snapshot shows:
  - Tasks (with status badges: OPEN/CLAIMED/DONE)
  - Artifacts produced so far (compact cards with type, author, version)
  - Signals raised (CONFLICT, UNCERTAINTY, etc.)

This is the GAIA equivalent of AgentVerse's blackboard visualization but
extracted from real run logs rather than synthetic mock-ups.

Usage:
  python experiments/viz/visualize_blackboard_trace.py \\
    --log-file experiments/humaneval/results/logs/HumanEval_0.jsonl \\
    --iterations 1 3 5 7 \\
    --out experiments/viz/figures/bb_trace_humaneval0.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


STATUS_COLORS = {
    "OPEN":    "#bdbdbd",
    "CLAIMED": "#ffeb3b",
    "DONE":    "#4caf50",
    "FAILED":  "#f44336",
    "BLOCKED": "#ff9800",
}

ARTIFACT_COLORS = {
    "CODE":          "#1976d2",
    "TEST_RESULT":   "#7b1fa2",
    "PLAN":          "#388e3c",
    "REVIEW":        "#f57c00",
    "DOCUMENTATION": "#5d4037",
}

SIGNAL_COLORS = {
    "CONFLICT":    "#d32f2f",
    "UNCERTAINTY": "#fbc02d",
    "DUPLICATION": "#1976d2",
    "URGENCY":     "#e91e63",
    "STALENESS":   "#9e9e9e",
    "BUDGET_RISK": "#ff5722",
}


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


def replay_to_iteration(events, target_iteration: int):
    """Walk events until iteration_start with iteration == target_iteration.

    Returns (tasks, artifacts, signals) — dictionaries representing the BB state.
    """
    tasks = {}        # task_id -> {title, status, type}
    artifacts = []    # list of {artifact_id, type, author, version, task_id}
    signals = []      # list of {type, description}

    current_iter = 0
    for evt in events:
        etype = evt.get("event_type") or evt.get("event") or ""
        details = evt.get("details", {}) or {}

        if etype == "iteration_start":
            current_iter = details.get("iteration", current_iter + 1)
            if current_iter > target_iteration:
                break

        if etype == "task_posted":
            tid = details.get("task_id")
            tasks[tid] = {
                "task_id": tid,
                "title": details.get("title", "")[:30],
                "status": details.get("status", "OPEN"),
                "type": details.get("task_type", "?"),
            }
        elif etype == "task_claimed":
            tid = details.get("task_id")
            if tid in tasks:
                tasks[tid]["status"] = "CLAIMED"
        elif etype == "task_completed":
            tid = details.get("task_id")
            if tid in tasks:
                tasks[tid]["status"] = "DONE"
        elif etype == "task_failed":
            tid = details.get("task_id")
            if tid in tasks:
                tasks[tid]["status"] = "FAILED"
        elif etype == "artifact_posted":
            artifacts.append({
                "artifact_id": details.get("artifact_id", "?")[:8],
                "type": details.get("type", "?"),
                "task_id": details.get("task_id", "?")[:8],
                "version": details.get("version", 1),
            })
        elif etype == "signal_posted":
            signals.append({
                "type": details.get("type", details.get("signal_type", "?")),
                "description": (details.get("description") or "")[:50],
                "severity": details.get("severity", 0.5),
            })

    return tasks, artifacts, signals


def draw_panel(ax, tasks, artifacts, signals, iter_num):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_axis_off()
    ax.set_title(f"Iteration {iter_num}", fontsize=11, fontweight="bold")

    # Tasks at the top
    ax.text(0.2, 11.4, f"Tasks ({len(tasks)})", fontsize=9, fontweight="bold")
    y = 10.9
    for t in list(tasks.values())[:6]:
        color = STATUS_COLORS.get(t["status"], "#aaaaaa")
        ax.add_patch(patches.Rectangle((0.2, y - 0.35), 9.5, 0.35,
                                         facecolor=color, edgecolor="black", linewidth=0.5))
        label = f"[{t['status'][:3]}] {t['title']}"
        ax.text(0.4, y - 0.18, label, fontsize=7, va="center")
        y -= 0.45
    if len(tasks) > 6:
        ax.text(0.4, y - 0.1, f"... +{len(tasks) - 6} more", fontsize=7, style="italic")

    # Artifacts in the middle
    ax.text(0.2, 7.4, f"Artifacts ({len(artifacts)})", fontsize=9, fontweight="bold")
    y = 6.9
    for a in artifacts[-6:]:
        color = ARTIFACT_COLORS.get(a["type"], "#888888")
        ax.add_patch(patches.Rectangle((0.2, y - 0.35), 9.5, 0.35,
                                         facecolor=color, edgecolor="black",
                                         linewidth=0.5, alpha=0.7))
        label = f"{a['type']} v{a['version']} (id={a['artifact_id']})"
        ax.text(0.4, y - 0.18, label, fontsize=7, va="center", color="white")
        y -= 0.45

    # Signals at the bottom
    ax.text(0.2, 3.4, f"Signals ({len(signals)})", fontsize=9, fontweight="bold")
    y = 2.9
    for s in signals[-5:]:
        color = SIGNAL_COLORS.get(s["type"], "#666666")
        ax.add_patch(patches.Rectangle((0.2, y - 0.35), 9.5, 0.35,
                                         facecolor=color, edgecolor="black",
                                         linewidth=0.5, alpha=0.8))
        label = f"[{s['type']}] {s['description']}"
        ax.text(0.4, y - 0.18, label, fontsize=7, va="center", color="white")
        y -= 0.45


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 3, 5, 7])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    events = parse_events(log_path)
    print(f"Loaded {len(events)} events from {log_path.name}")

    n = len(args.iterations)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, it in zip(axes, args.iterations):
        tasks, artifacts, signals = replay_to_iteration(events, it)
        draw_panel(ax, tasks, artifacts, signals, it)

    fig.suptitle(f"Blackboard Trace — {log_path.stem}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
