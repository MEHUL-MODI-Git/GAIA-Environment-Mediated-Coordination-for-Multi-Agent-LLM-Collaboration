"""Regenerate fig1_architecture.png with larger text and clean layout."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "poster_figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE        = "#1565C0"
GREEN       = "#2E7D32"
ORANGE      = "#E65100"
RED         = "#B71C1C"
PURPLE      = "#6A1B9A"
GREY        = "#455A64"
LIGHT_BLUE  = "#BBDEFB"
LIGHT_GREEN = "#C8E6C9"

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('white')

def rbox(x, y, w, h, label, sublabel="", fc="white", ec=BLUE, lw=2.5,
         fontsize=13, tc=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(rect)
    if tc is None:
        tc = "white" if fc not in ("white", LIGHT_BLUE, LIGHT_GREEN,
                                    "#FFE0B2", "#FFCDD2", "#E1BEE7",
                                    "#ECEFF1", "#E8EAF6", "#FFF9C4",
                                    "#F3E5F5", "#FFF3E0") else "black"
    ax.text(x + w/2, y + h/2 + (0.25 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=tc, zorder=4, linespacing=1.3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.35, sublabel,
                ha="center", va="center", fontsize=fontsize - 3,
                color="#555" if tc == "black" else "#DDD", zorder=4)

def arr(x1, y1, x2, y2, color=GREY, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                               mutation_scale=18), zorder=5)

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(11, 12.55, "GAIA: Multi-Agent Blackboard Architecture",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#1A237E")
ax.text(11, 12.05, "Shared Blackboard  +  Verification Gate  +  Conflict-as-Task  +  Branch-and-Merge",
        ha="center", va="center", fontsize=12, color=GREY)

# ═══════════════════════════════════════════════════════════════
# PHASE COLUMN HEADERS
# ═══════════════════════════════════════════════════════════════
headers = [
    (0.3,  3.5,  "Phase 1\nParallel Solving",        BLUE),
    (5.8,  1.5,  "Phase 2\nAggregation",             ORANGE),
    (9.2,  4.5,  "Shared Blackboard\n(World State)", "#3949AB"),
    (15.5, 1.5,  "Phase 3\nReconciliation",          RED),
    (18.8, 1.5,  "Phase 4\nVerification",            PURPLE),
]
for px, pw, pl, pc in headers:
    ax.text(px + pw/2, 11.55, pl, ha="center", va="center",
            fontsize=13, color=pc, fontweight="bold", linespacing=1.4)

# ═══════════════════════════════════════════════════════════════
# SHARED BLACKBOARD (centre)
# ═══════════════════════════════════════════════════════════════
bb = FancyBboxPatch((9.0, 3.2), 6.5, 7.8, boxstyle="round,pad=0.25",
                     fc="#E8EAF6", ec="#3949AB", lw=3, zorder=1, alpha=0.9)
ax.add_patch(bb)
ax.text(12.25, 10.75, "Shared Blackboard", ha="center", fontsize=14,
        fontweight="bold", color="#1A237E", zorder=4)

# Artifact cards on blackboard
art = [
    ("TASK",       9.3,  9.4,  2.6, 1.0, "#1A237E"),
    ("PLAN ×3",    9.3,  8.1,  2.6, 1.0, "#283593"),
    ("⚡ CONFLICT", 9.3,  6.5,  2.6, 1.1, "#B71C1C"),
    ("REVIEW",     12.4, 9.4,  2.6, 1.0, "#1B5E20"),
    ("EVIDENCE",   12.4, 8.1,  2.6, 1.0, "#4A148C"),
]
for label, x, y, w, h, fc in art:
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                        fc=fc, ec="white", lw=2, zorder=3)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=4)

# CONFLICT → triggers label
arr(10.6, 6.5, 10.6, 5.6, color=RED, lw=1.8)
ax.text(10.85, 6.0, "triggers", fontsize=10, color=RED, fontstyle="italic", zorder=5)

# Blackboard primitives list
primitives = [
    "Tasks  (status, priority, leasing)",
    "Artifacts  (plans, reviews, code)",
    "Claims  (assertions + confidence)",
    "Evidence  (test logs, pass/fail)",
    "Signals  (conflict, uncertainty…)",
    "Audit Log  (full trace)",
]
for i, p in enumerate(primitives):
    ax.text(9.3, 5.1 - i * 0.36, f"• {p}", fontsize=9.5, color="#333",
            va="center", zorder=4)

# ═══════════════════════════════════════════════════════════════
# SOLVERS (left)
# ═══════════════════════════════════════════════════════════════
solver_y     = [9.4, 7.9, 6.4]
temps        = ["temp = 0.0", "temp = 0.3", "temp = 0.6"]
solver_cols  = [BLUE, BLUE, "#C62828"]

for i, (sy, t, col) in enumerate(zip(solver_y, temps, solver_cols)):
    rbox(0.3, sy, 3.5, 1.1, f"Solver {i+1}", t, fc=col, ec=col, fontsize=14)
    # TASK → Solver
    arr(9.3, 9.9, 3.8, sy + 0.55, color=GREY, lw=1.4)
    # Solver → PLAN
    arr(3.8, sy + 0.55, 9.3, 8.6, color=BLUE if col == BLUE else "#C62828", lw=1.6)

# ═══════════════════════════════════════════════════════════════
# AGGREGATOR
# ═══════════════════════════════════════════════════════════════
rbox(5.2, 7.4, 3.5, 1.3, "Aggregator", "Reads all PLANs → detects conflict",
     fc=ORANGE, ec=ORANGE, fontsize=14)
arr(9.3, 8.6, 8.7, 8.05, color=ORANGE, lw=1.8)    # PLAN → Aggregator
arr(8.7, 7.4, 9.3, 7.05, color=ORANGE, lw=1.8)    # Aggregator → CONFLICT
ax.text(7.0, 6.7, "If conflict →", fontsize=12, color=RED,
        fontstyle="italic", fontweight="bold", zorder=5)

# ═══════════════════════════════════════════════════════════════
# RECONCILER
# ═══════════════════════════════════════════════════════════════
rbox(15.5, 5.8, 4.0, 1.8, "Reconciler\n(GPT-4.1)",
     "Conditional on CONFLICT only", fc=RED, ec=RED, fontsize=14)
arr(11.9, 6.5, 15.5, 6.7, color=RED, lw=2.0)       # CONFLICT → Reconciler
arr(15.5, 7.4, 15.0, 9.9, color=GREEN, lw=2.0)     # Reconciler → REVIEW

# ═══════════════════════════════════════════════════════════════
# VERIFIER
# ═══════════════════════════════════════════════════════════════
rbox(18.5, 8.6, 3.2, 1.3, "Verifier", "Ground-truth check", fc=PURPLE, ec=PURPLE, fontsize=14)
arr(15.0, 9.9, 18.5, 9.25, color=PURPLE, lw=2.0)   # REVIEW → Verifier
arr(18.5, 8.6, 15.0, 8.6,  color=PURPLE, lw=1.6)   # EVIDENCE ←

# ═══════════════════════════════════════════════════════════════
# ITERATE LOOP annotation
# ═══════════════════════════════════════════════════════════════
ax.annotate("", xy=(0.15, 7.9), xytext=(0.15, 10.9),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5,
                           connectionstyle="arc3,rad=-0.35"), zorder=5)
ax.text(0.05, 9.4, "iterate\nuntil all\ntasks done",
        ha="center", va="center", fontsize=10, color=BLUE,
        fontweight="bold", linespacing=1.3, rotation=90)

# ═══════════════════════════════════════════════════════════════
# BOTTOM NOTES — Branch-and-Merge  |  Conflict-as-Task
# ═══════════════════════════════════════════════════════════════
# Branch-and-Merge box
bm = FancyBboxPatch((0.3, 0.4), 10.0, 2.2, boxstyle="round,pad=0.2",
                     fc="#F3E5F5", ec=PURPLE, lw=2.2, zorder=2)
ax.add_patch(bm)
ax.text(5.3, 2.3, "Branch-and-Merge", ha="center", fontsize=14,
        fontweight="bold", color="#4A148C", zorder=4)
ax.text(5.3, 1.7, "On uncertainty → fork to mini-blackboards (Plan A / Plan B)", ha="center",
        fontsize=12, color="#4A148C", zorder=4)
ax.text(5.3, 1.2, "Each branch runs independently with its own agent pool", ha="center",
        fontsize=12, color="#4A148C", zorder=4)
ax.text(5.3, 0.7, "Best verified result merged back to main blackboard", ha="center",
        fontsize=12, color="#4A148C", zorder=4)

# Conflict-as-Task box
ct = FancyBboxPatch((11.0, 0.4), 10.7, 2.2, boxstyle="round,pad=0.2",
                     fc="#FFF3E0", ec=ORANGE, lw=2.2, zorder=2)
ax.add_patch(ct)
ax.text(16.35, 2.3, "Conflict-as-Task", ha="center", fontsize=14,
        fontweight="bold", color="#BF360C", zorder=4)
ax.text(16.35, 1.7, "Failed or conflicting results are not discarded", ha="center",
        fontsize=12, color="#BF360C", zorder=4)
ax.text(16.35, 1.2, "Re-posted as a new TASK on the blackboard with failure feedback", ha="center",
        fontsize=12, color="#BF360C", zorder=4)
ax.text(16.35, 0.7, "Agents self-claim and retry with additional context", ha="center",
        fontsize=12, color="#BF360C", zorder=4)

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(fc=BLUE,      ec=BLUE,      label="Solver Agents (fast model)"),
    mpatches.Patch(fc=ORANGE,    ec=ORANGE,    label="Aggregator Agent"),
    mpatches.Patch(fc=RED,       ec=RED,       label="Reconciler (capable model, conditional)"),
    mpatches.Patch(fc=PURPLE,    ec=PURPLE,    label="Verifier Agent"),
    mpatches.Patch(fc="#E8EAF6", ec="#3949AB", lw=2.5, label="Shared Blackboard (typed primitives)"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=11,
          framealpha=0.95, ncol=1, bbox_to_anchor=(1.0, 0.98))

fig.tight_layout(pad=0.4)
path = os.path.join(OUT_DIR, "fig1_architecture.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved {path}")
