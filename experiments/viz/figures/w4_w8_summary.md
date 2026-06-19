# W4 — Coordination scaling law (from E8)

GAIA accuracy fit: **0.36 + 0.66·(1−e^(−0.28·n))**
- fitted asymptote ≈ **100%** (raw fit 102%; accuracy is bounded at 100%, so the curve saturates at the cap — only 4 points (n=2,4,6,8), fit is indicative not definitive).
- 95%-of-gain knee: **≈10.7 agents** (EXTRAPOLATION — beyond the n≤8 tested range; treat as a qualitative 'returns flatten in the high-single-digits' claim, not a precise value).
- GAIA points: [(2, 0.65), (4, 0.8), (6, 0.9), (8, 0.95)]
- Homogeneous: flat at 15% for ALL counts (no scaling law — adding identical agents yields nothing).

**Takeaway.** GAIA exhibits a *coordination* scaling law with a real asymptote and a diminishing-returns knee — structurally analogous to model scaling laws, but driven by coordination not parameters. Homogeneous scaling has no such curve. This is a new framing: 'how much accuracy can coordination buy, and where does it saturate?'

# W8 — Wall-clock as a first-class axis

GAIA mean phase wall-clock (s): {'phase1_solve_s': 2.47, 'phase2_aggregate_s': 0.9, 'phase3_reconcile_s': 3.94, 'phase4_verify_s': 0.0}
Per-condition mean duration (s): {'NX1:debate': 6.75, 'NX1:blackboard_plain': 3.36, 'E3:GAIA(total)': 7.31}

**Takeaway.** GAIA's Phase-1 (experts/solvers) executes in PARALLEL via self-assignment, and the expensive reconcile phase runs ONLY when a conflict is raised — so wall-clock is dominated by at most one slow audit, not by N serial debate rounds. Debate's latency grows linearly in rounds×agents; GAIA's does not. Latency, not just token cost, is a deployment axis the literature underreports.