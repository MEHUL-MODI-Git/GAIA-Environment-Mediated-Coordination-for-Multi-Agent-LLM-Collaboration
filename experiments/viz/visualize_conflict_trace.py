#!/usr/bin/env python3
"""Render a horizontal conflict flow diagram from E3 or E9 results.

Shows: agent outputs → conflict signal → reconciler/auditor diagnosis → final answer.
Wrong agents in red, correct path in green.

Usage:
  python experiments/viz/visualize_conflict_trace.py \\
    --results experiments/correlated_failure/results/correlated_failure_<ts>.json \\
    --problem-id trap_rate_001 \\
    --out experiments/viz/figures/conflict_trace_rate_001.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def find_e3_record(results: dict, problem_id: str):
    """Pull the GAIA-condition record for the given problem_id."""
    if "gaia" in results:
        for r in results["gaia"]["results"]:
            if r["problem_id"] == problem_id:
                return r
    return None


def find_e9_record(results: dict, puzzle_id: str):
    """Pull the fault_gaia record for the given puzzle_id."""
    if "fault_gaia" in results:
        for r in results["fault_gaia"]["results"]:
            if r["puzzle_id"] == puzzle_id:
                return r
    return None


def draw_box(ax, x, y, w, h, text, color, edge="black", text_color="black", fontsize=9):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    ))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=text_color, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="black", width=2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, linewidth=width),
    )


def render_e3_trace(record: dict, problem_id: str, out_path: Path):
    truth = record.get("ground_truth")
    proposed = record.get("proposed_answer")
    trap_aware = record.get("trap_aware_answer")
    standard = record.get("standard_solver_answers", {})
    common_wrong = record.get("common_wrong_answer")
    category = record.get("trap_category", "?")
    reconciled = record.get("reconciler_sided_with_trap_aware")
    passed = record.get("passed")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Column 1: 3 solvers
    ax.text(1.5, 7.4, "Phase 1: Solvers", fontsize=11, fontweight="bold", ha="center")
    solvers = list(standard.items()) + [("TrapAware", trap_aware)]
    y_solvers = [6.0, 4.5, 3.0]
    for (name, ans), y in zip(solvers, y_solvers):
        is_correct = (ans == truth)
        color = "#c8e6c9" if is_correct else "#ffcdd2"
        draw_box(ax, 0.3, y - 0.5, 2.5, 1.0,
                 f"{name}\nans={ans}", color)

    # Column 2: Aggregator/conflict
    ax.text(5, 7.4, "Phase 2: Aggregator", fontsize=11, fontweight="bold", ha="center")
    draw_box(ax, 4, 4.0, 2, 1.5, "CONFLICT\ndetected", "#fff9c4", edge="#d32f2f")

    # Column 3: Reconciler
    ax.text(8.5, 7.4, "Phase 3: Reconciler", fontsize=11, fontweight="bold", ha="center")
    rec_color = "#c8e6c9" if reconciled else "#ffe0b2"
    rec_text = "Sided with\ntrap-aware\n(correct!)" if reconciled else "Sided with\nmajority"
    draw_box(ax, 7.5, 4.0, 2, 1.5, rec_text, rec_color)

    # Column 4: Final
    ax.text(12, 7.4, "Phase 4: Verifier", fontsize=11, fontweight="bold", ha="center")
    final_color = "#c8e6c9" if passed else "#ffcdd2"
    final_text = f"Final: {proposed}\n{'PASS' if passed else 'FAIL'}\n(truth={truth})"
    draw_box(ax, 11, 4.0, 2, 1.5, final_text, final_color)

    # Arrows
    for y in y_solvers:
        draw_arrow(ax, 2.85, y, 4.0, 4.75)
    draw_arrow(ax, 6.05, 4.75, 7.5, 4.75)
    draw_arrow(ax, 9.55, 4.75, 11.0, 4.75)

    fig.suptitle(
        f"E3 Conflict Trace — {problem_id} ({category} trap, common wrong: {common_wrong})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def render_e9_trace(record: dict, puzzle_id: str, out_path: Path):
    """Render the fault-injection conflict trace.

    Shows: 4 experts → DeductionAuditor (trust scores) → TrustAwareSynth → Final.
    Highlights the faulty expert in red, the auditor's correct flag in green.
    """
    trust_scores = record.get("trust_scores", {})
    real_faulty = record.get("real_faulty_agent_id")
    auditor_suspect = record.get("auditor_suspect_id")
    flagged_correctly = record.get("auditor_flagged_faulty_agent")
    passed = record.get("passed")
    n_contradictions = record.get("n_contradictions_found", 0)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Column 1: Experts with trust scores
    ax.text(1.5, 7.4, "Phase 1: Experts", fontsize=11, fontweight="bold", ha="center")
    y_positions = [6.5, 5.0, 3.5, 2.0]
    items = list(trust_scores.items())[:4]
    for (aid, score), y in zip(items, y_positions):
        is_faulty = (aid == real_faulty)
        # Color by trust score
        if score >= 0.7:
            color = "#c8e6c9"
        elif score >= 0.4:
            color = "#fff9c4"
        else:
            color = "#ffcdd2"
        label = f"Agent {aid[:6]}\ntrust={score:.2f}"
        if is_faulty:
            label = "FAULTY\n" + label
        edge = "red" if is_faulty else "black"
        draw_box(ax, 0.3, y - 0.5, 2.5, 1.0, label, color, edge=edge)

    # Column 2: Auditor
    ax.text(5, 7.4, "Phase 1b: Auditor", fontsize=11, fontweight="bold", ha="center")
    audit_text = f"Found {n_contradictions}\ncontradictions\nFlagged: "
    if auditor_suspect:
        audit_text += f"{auditor_suspect[:6]}"
        if flagged_correctly:
            audit_text += "\n(CORRECT!)"
        else:
            audit_text += "\n(MISIDENTIFIED)"
    else:
        audit_text += "NONE"
    audit_color = "#c8e6c9" if flagged_correctly else "#fff9c4"
    draw_box(ax, 4, 4.0, 2, 1.5, audit_text, audit_color)

    # Column 3: Trust-aware synthesizer
    ax.text(8.5, 7.4, "Phase 2: TrustAware Synth", fontsize=11, fontweight="bold", ha="center")
    draw_box(ax, 7.5, 4.0, 2, 1.5, "Downweights\nlow-trust\nagents", "#e1f5fe")

    # Column 4: Final
    ax.text(12, 7.4, "Phase 4: Verifier", fontsize=11, fontweight="bold", ha="center")
    final_color = "#c8e6c9" if passed else "#ffcdd2"
    final_text = f"{'PASS' if passed else 'FAIL'}"
    draw_box(ax, 11, 4.0, 2, 1.5, final_text, final_color)

    # Arrows
    for y in y_positions:
        draw_arrow(ax, 2.85, y, 4.0, 4.75)
    draw_arrow(ax, 6.05, 4.75, 7.5, 4.75)
    draw_arrow(ax, 9.55, 4.75, 11.0, 4.75)

    fig.suptitle(
        f"E9 Conflict Trace — {puzzle_id}  "
        f"(faulty agent: {real_faulty[:6] if real_faulty else '?'}, "
        f"auditor flagged correctly: {flagged_correctly})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to results JSON (E3 or E9)")
    parser.add_argument("--problem-id", required=True,
                        help="The problem_id or puzzle_id to render")
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment", choices=["e3", "e9"], default=None,
                        help="Experiment type (auto-detected if omitted)")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    exp = args.experiment
    if exp is None:
        if "gaia" in data:
            exp = "e3"
        elif "fault_gaia" in data:
            exp = "e9"
        else:
            print(f"Could not auto-detect experiment type from keys: {list(data.keys())}")
            return

    if exp == "e3":
        rec = find_e3_record(data, args.problem_id)
        if rec is None:
            print(f"Problem {args.problem_id} not found in GAIA results")
            return
        render_e3_trace(rec, args.problem_id, Path(args.out))
    else:
        rec = find_e9_record(data, args.problem_id)
        if rec is None:
            print(f"Puzzle {args.problem_id} not found in fault_gaia results")
            return
        render_e9_trace(rec, args.problem_id, Path(args.out))


if __name__ == "__main__":
    main()
