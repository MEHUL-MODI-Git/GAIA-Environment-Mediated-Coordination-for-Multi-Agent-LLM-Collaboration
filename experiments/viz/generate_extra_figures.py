"""
Generate extra poster figures A-D for GAIA URECA poster.
Output: experiments/viz/poster_figures/fig_A/B/C/D.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "poster_figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE        = "#1565C0"
GREEN       = "#2E7D32"
ORANGE      = "#E65100"
RED         = "#B71C1C"
PURPLE      = "#6A1B9A"
GREY        = "#455A64"
TEAL        = "#00695C"
LIGHT_BLUE  = "#BBDEFB"
LIGHT_GREEN = "#C8E6C9"
LIGHT_ORANGE= "#FFE0B2"
LIGHT_RED   = "#FFCDD2"
LIGHT_PURPLE= "#E1BEE7"
LIGHT_GREY  = "#ECEFF1"
INDIGO      = "#283593"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Blackboard State Timeline (board filling up during one episode)
# ══════════════════════════════════════════════════════════════════════════════
def draw_figure_A():
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    states = [
        {
            "title": "t=0\nProblem Posted",
            "cards": [
                ("TASK", "#1A237E", "white", "Snail climbing\n30-ft pole with\nrainy days"),
            ],
            "title_color": GREY,
        },
        {
            "title": "t=1\n3 Solvers Running",
            "cards": [
                ("TASK", "#1A237E", "white", "Snail problem"),
                ("PLAN 1", BLUE, "white", "Answer: 11\nTemp=0.0"),
                ("PLAN 2", BLUE, "white", "Answer: 11\nTemp=0.3"),
                ("PLAN 3", "#C62828", "white", "Answer: 13\nTemp=0.6 ✗"),
            ],
            "title_color": BLUE,
        },
        {
            "title": "t=2\nConflict Detected",
            "cards": [
                ("TASK", "#1A237E", "white", "Snail problem"),
                ("PLAN ×3", BLUE, "white", "2 say 11\n1 says 13"),
                ("⚡ CONFLICT", "#B71C1C", "white", "Disagreement\ndetected!"),
            ],
            "title_color": RED,
        },
        {
            "title": "t=3\nReconciler Resolves",
            "cards": [
                ("TASK", "#1A237E", "white", "Snail problem"),
                ("PLAN ×3", BLUE, "white", "2 say 11\n1 says 13"),
                ("⚡ CONFLICT", "#B71C1C", "white", "resolved ✓"),
                ("REVIEW", "#1B5E20", "white", "Final: 11\nSolver 3 had\nday-9 error"),
            ],
            "title_color": GREEN,
        },
        {
            "title": "t=4\nVerified & Done",
            "cards": [
                ("TASK", "#1A237E", "white", "Snail problem"),
                ("PLAN ×3", BLUE, "white", "logged"),
                ("CONFLICT", "#555", "white", "resolved"),
                ("REVIEW", "#1B5E20", "white", "Answer: 11"),
                ("✓ EVIDENCE", "#4A148C", "white", "passed=True\n11 == 11"),
            ],
            "title_color": PURPLE,
        },
    ]

    for ax, state in zip(axes, states):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6.5)
        ax.axis("off")
        ax.set_facecolor("#F8F9FA")

        # Board background
        board = FancyBboxPatch((0.1, 0.2), 3.8, 5.9,
                               boxstyle="round,pad=0.1",
                               fc="#FAFAFA", ec="#90CAF9", lw=2, zorder=1)
        ax.add_patch(board)

        # Title
        ax.text(2.0, 6.1, state["title"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=state["title_color"],
                linespacing=1.3)

        # Cards stacked vertically
        card_h = 0.85
        start_y = 5.1
        for label, fc, tc, content in state["cards"]:
            card = FancyBboxPatch((0.3, start_y - card_h + 0.05), 3.4, card_h - 0.1,
                                  boxstyle="round,pad=0.07",
                                  fc=fc, ec="white", lw=1.5, zorder=3)
            ax.add_patch(card)
            ax.text(2.0, start_y - card_h/2 + 0.05, f"{label}\n{content}",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=tc,
                    linespacing=1.25, zorder=4)
            start_y -= card_h + 0.08

    # Arrow between panels
    for i in range(4):
        axes[i].annotate("", xy=(1.0, 0.5), xytext=(-0.3, 0.5),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.0))

    fig.suptitle("Blackboard State Evolution — One Episode (GSM8K Conflict Example)",
                 fontsize=13, fontweight="bold", color="#1A237E", y=1.02)
    fig.tight_layout(pad=0.8)
    path = os.path.join(OUT_DIR, "figA_blackboard_timeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Silent Failure vs GAIA (side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
def draw_figure_B():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    def rbox(ax, x, y, w, h, fc, ec, label, sublabel="", fontsize=10, lw=2, tc=None):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(r)
        if tc is None:
            tc = "white" if fc not in ("#F5F5F5", LIGHT_GREEN, LIGHT_RED,
                                        LIGHT_BLUE, LIGHT_ORANGE, "#FFF9C4", "white") else "black"
        ax.text(x+w/2, y+h/2+(0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=tc, zorder=4)
        if sublabel:
            ax.text(x+w/2, y+h/2-0.22, sublabel, ha="center", va="center",
                    fontsize=8, color="#999" if tc=="white" else "#666", zorder=4)

    def arr(ax, x1, y1, x2, y2, color=GREY, lw=1.8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   mutation_scale=14), zorder=5)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis("off")

    # ── LEFT: Single Agent ──────────────────────────────────────────────────
    ax1.set_facecolor("#FFF8F8")
    ax1.text(3.0, 7.6, "Single Agent", ha="center", fontsize=14,
             fontweight="bold", color=RED)
    ax1.text(3.0, 7.1, "One attempt. No verification. No recovery.",
             ha="center", fontsize=9.5, color=GREY)

    # Problem
    rbox(ax1, 1.5, 5.8, 3.0, 0.9, "#E8EAF6", "#3949AB",
         "PROBLEM", "Hard math word problem", fontsize=10)

    # Single agent thinks
    rbox(ax1, 1.5, 4.3, 3.0, 1.0, GREY, GREY, "Single Agent",
         "Reasons step by step...", fontsize=10)
    arr(ax1, 3.0, 5.8, 3.0, 5.3, GREY)

    # Wrong output with confidence
    rbox(ax1, 1.5, 2.9, 3.0, 0.9, "#C62828", "#C62828",
         "Output: 2", "Confidently wrong (truth = 44)", fontsize=10)
    arr(ax1, 3.0, 4.3, 3.0, 3.8, RED)

    # No check → accepted
    rbox(ax1, 1.5, 1.5, 3.0, 0.85, LIGHT_RED, RED,
         "✗ WRONG ANSWER ACCEPTED", "No flag raised. No retry.", fontsize=9.5, tc="black")
    arr(ax1, 3.0, 2.9, 3.0, 2.35, RED)

    # Danger label
    ax1.text(3.0, 0.85, "Silent failure — undetectable",
             ha="center", fontsize=10, color=RED, fontweight="bold",
             bbox=dict(fc=LIGHT_RED, ec=RED, lw=1.5, boxstyle="round,pad=0.3"))

    # ── RIGHT: GAIA ──────────────────────────────────────────────────────────
    ax2.set_facecolor("#F8FFF8")
    ax2.text(3.0, 7.6, "GAIA", ha="center", fontsize=14,
             fontweight="bold", color=GREEN)
    ax2.text(3.0, 7.1, "3 agents. Conflict detection. Evidence-based fix.",
             ha="center", fontsize=9.5, color=GREY)

    # Problem
    rbox(ax2, 1.5, 5.8, 3.0, 0.9, "#E8EAF6", "#3949AB",
         "PROBLEM", "Same hard math word problem", fontsize=10)

    # 3 solvers
    rbox(ax2, 0.2, 4.5, 1.5, 0.75, BLUE, BLUE, "Solver 1", "→ 44 ✓", fontsize=9)
    rbox(ax2, 2.25, 4.5, 1.5, 0.75, BLUE, BLUE, "Solver 2", "→ 44 ✓", fontsize=9)
    rbox(ax2, 4.3, 4.5, 1.5, 0.75, "#C62828", "#C62828", "Solver 3", "→ 2 ✗", fontsize=9)
    arr(ax2, 3.0, 5.8, 1.0, 5.25, BLUE)
    arr(ax2, 3.0, 5.8, 3.0, 5.25, BLUE)
    arr(ax2, 3.0, 5.8, 5.05, 5.25, RED)

    # Aggregator catches conflict
    rbox(ax2, 1.5, 3.2, 3.0, 0.85, ORANGE, ORANGE,
         "Aggregator", "Conflict detected: 2 vs 1", fontsize=9.5)
    arr(ax2, 1.0, 4.5, 2.0, 4.05, ORANGE)
    arr(ax2, 3.0, 4.5, 3.0, 4.05, ORANGE)
    arr(ax2, 5.05, 4.5, 4.0, 4.05, ORANGE)

    # Reconciler
    rbox(ax2, 1.5, 2.0, 3.0, 0.85, "#7B1FA2", "#7B1FA2",
         "Reconciler (GPT-4.1)", "Finds Solver-3 error → corrects", fontsize=9)
    arr(ax2, 3.0, 3.2, 3.0, 2.85, PURPLE)

    # Verified result
    rbox(ax2, 1.5, 0.75, 3.0, 0.85, LIGHT_GREEN, GREEN,
         "✓ VERIFIED: 44  PASS", "Conflict trace logged", fontsize=9.5, tc="black")
    arr(ax2, 3.0, 2.0, 3.0, 1.6, GREEN)

    fig.suptitle("Silent Failure vs. GAIA — Same Problem, Different Outcomes",
                 fontsize=13, fontweight="bold", color="#1A237E", y=1.01)
    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, "figB_silent_failure.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Cost vs Accuracy Bubble Chart
# ══════════════════════════════════════════════════════════════════════════════
def draw_figure_C():
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    # Data points: (cost_per_problem_usd, accuracy_pct, auditability_0_to_3, label, color)
    points = [
        (0.0045, 95.0,  0, "Single Agent\n(GSM8K)",    GREY),
        (0.0127, 100.0, 1, "Majority Vote\n(GSM8K)",   LIGHT_BLUE),
        (0.0160, 100.0, 3, "GAIA\n(GSM8K)",            BLUE),
        (0.0010, 58.5,  0, "Single Agent\n(HumanEval)", GREY),
        (0.0020, 96.4,  3, "GAIA\n(HumanEval)",         BLUE),
        (0.0590, 95.0,  3, "GAIA\n(Logic Puzzle)",       "#7B1FA2"),
        (0.0200, 10.0,  0, "Isolated\n(Logic Puzzle)",   LIGHT_RED),
    ]

    for cost, acc, audit, label, color in points:
        size = 120 + audit * 280   # bubble size scales with auditability
        ec = "#333333" if color in (LIGHT_BLUE, LIGHT_RED) else color
        ax.scatter(cost * 1000, acc, s=size, color=color,
                   edgecolors=ec, linewidths=2, zorder=4, alpha=0.88)
        # Label offset to avoid overlap
        offsets = {
            "Single Agent\n(GSM8K)":    (-0.6,  -6),
            "Majority Vote\n(GSM8K)":   ( 0.3,   2),
            "GAIA\n(GSM8K)":            ( 0.3,  -6),
            "Single Agent\n(HumanEval)":(-0.6,   2),
            "GAIA\n(HumanEval)":        ( 0.3,   2),
            "GAIA\n(Logic Puzzle)":     ( 0.3,   2),
            "Isolated\n(Logic Puzzle)": ( 0.3,  -7),
        }
        dx, dy = offsets.get(label, (0.2, 2))
        ax.text(cost * 1000 + dx, acc + dy, label,
                ha="left" if dx >= 0 else "right",
                va="center", fontsize=8.5, color="#333333",
                fontweight="bold", linespacing=1.3)

    # Auditability legend (bubble size)
    for audit_level, audit_label in [(0, "None"), (1, "Partial"), (3, "Full")]:
        size = 120 + audit_level * 280
        ax.scatter([], [], s=size, color=GREY, alpha=0.7,
                   label=f"Auditability: {audit_label}")

    # Arrow annotation: GAIA value proposition
    ax.annotate("GAIA: higher accuracy\n+ full audit trail",
                xy=(16, 96.4), xytext=(30, 85),
                fontsize=9.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8),
                bbox=dict(fc=LIGHT_BLUE, ec=BLUE, lw=1.2,
                          boxstyle="round,pad=0.35"))

    ax.set_xlabel("Cost per Problem (USD × 10⁻³)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Cost–Accuracy–Auditability Tradeoff\n(bubble size = auditability level)",
                 fontsize=13, fontweight="bold", color="#1A237E", pad=10)
    ax.set_xlim(-2, 75)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9,
              title="Bubble size", title_fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figC_cost_accuracy.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — Agent Gantt Chart (Phase Timeline)
# ══════════════════════════════════════════════════════════════════════════════
def draw_figure_D():
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("white")

    agents = [
        "Verifier",
        "Reconciler\n(GPT-4.1)",
        "Aggregator",
        "Solver 3\n(T=0.6)",
        "Solver 2\n(T=0.3)",
        "Solver 1\n(T=0.0)",
    ]
    y_pos = list(range(len(agents)))

    # (start, end, color, label)
    bars = [
        # Solver 1 — fastest
        (0.0, 3.2,  BLUE,    "Solving"),
        # Solver 2
        (0.0, 4.5,  BLUE,    "Solving"),
        # Solver 3 — slowest, also has error
        (0.0, 5.1,  "#C62828","Solving (wrong)"),
        # Aggregator — fires after all solvers done
        (5.1, 5.75, ORANGE,  "Aggregation"),
        # Reconciler — only on conflict
        (5.75, 8.3, "#7B1FA2","Reconciliation\n(conflict only)"),
        # Verifier — shown clearly
        (8.3, 9.0,  GREEN,   "Verification"),
    ]

    bar_data = list(zip(y_pos, bars))
    for yi, (start, end, color, label) in bar_data:
        ax.barh(yi, end - start, left=start, height=0.55,
                color=color, edgecolor="white", linewidth=1.5,
                alpha=0.9, zorder=3)
        if label:
            mid = start + (end - start) / 2
            ax.text(mid, yi, label, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                    linespacing=1.2)

    # Vertical markers for key events
    events = [
        (0.0,  "#1A237E", "Problem\nPosted"),
        (5.1,  ORANGE,    "All PLANs\nReady"),
        (5.75, RED,       "CONFLICT\nSignal"),
        (8.3,  GREEN,     "REVIEW\nReady"),
        (9.0,  PURPLE,    "EVIDENCE\nPosted"),
    ]
    for x, color, label in events:
        ax.axvline(x, color=color, lw=1.8, linestyle="--", alpha=0.8, zorder=2)
        ax.text(x + 0.07, len(agents) - 0.1, label,
                fontsize=8, color=color, fontweight="bold",
                va="top", linespacing=1.3)

    # Phase background shading
    phase_regions = [
        (0.0,  5.1,  "#E3F2FD", "Phase 1\nParallel Solving"),
        (5.1,  5.75, "#FFF3E0", "Phase 2\nAggregation"),
        (5.75, 8.3,  "#FCE4EC", "Phase 3\nReconciliation"),
        (8.3,  9.5,  "#E8F5E9", "Phase 4\nVerification"),
    ]
    for x0, x1, fc, label in phase_regions:
        ax.axvspan(x0, x1, alpha=0.22, color=fc, zorder=0)
        ax.text((x0 + x1) / 2, -0.85, label, ha="center", va="top",
                fontsize=8.5, color=GREY, linespacing=1.25,
                fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agents, fontsize=10.5)
    ax.set_xlabel("Wall-clock time (seconds)", fontsize=12)
    ax.set_xlim(-0.3, 11.0)
    ax.set_ylim(-1.3, len(agents) + 0.5)
    ax.set_title("Agent Execution Timeline — GSM8K Conflict Episode\n"
                 "Solvers run in parallel (Phase 1); Reconciler only activates on conflict",
                 fontsize=12, fontweight="bold", color="#1A237E", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Highlight: parallel speedup annotation
    ax.annotate("3× parallel\n(vs sequential ~12.8s)",
                xy=(2.5, 2.5), xytext=(6.5, 4.3),
                fontsize=9, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5),
                bbox=dict(fc=LIGHT_BLUE, ec=BLUE, lw=1.2,
                          boxstyle="round,pad=0.3"))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "figD_gantt.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating extra poster figures A–D...")
    draw_figure_A()
    draw_figure_B()
    draw_figure_C()
    draw_figure_D()
    print(f"\nAll figures saved to: {OUT_DIR}")
