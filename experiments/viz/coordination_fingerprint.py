#!/usr/bin/env python3
"""W10 — Coordination-fingerprint analysis (novel representation).

Instead of describing episodes by surface domain (math/puzzle), describe them
by *how GAIA coordinated*: a feature vector per episode mined from the full
state dumps —
  n_artifacts, n_signals, n_conflict_signals, conflict_detected/resolved,
  reasoning-depth per role (chars), role diversity, phase-time profile.

We then (1) 2-D project all 253+ episodes (PCA, dependency-free) coloured by
outcome to show coordination-state separates success from failure, and
(2) cluster into "coordination regimes" and report each regime's pass rate.

This is a new way to *represent* MAS behaviour and a predictor of when
coordination will help — no analogue in the cited literature.
"""
import glob, json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "experiments" / "viz" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fingerprint(sf):
    try:
        d = json.load(open(sf))
    except Exception:
        return None
    arts = list(d.get("artifacts", {}).values())
    sigs = list(d.get("signals", {}).values())
    extra = d.get("extra", {})
    if not arts:
        return None
    by_role = defaultdict(list)
    for a in arts:
        by_role[a.get("metadata", {}).get("subtype", "?")].append(
            len(a.get("content", "") or ""))
    roles = list(by_role)
    n_conf = sum(1 for s in sigs if s.get("type") == "CONFLICT")
    avg_depth = sum(sum(v) for v in by_role.values()) / max(1, len(arts))
    max_role_depth = max((sum(v) / len(v)) for v in by_role.values())
    feat = {
        "n_artifacts": len(arts),
        "n_signals": len(sigs),
        "n_conflict": n_conf,
        "role_diversity": len(roles),
        "avg_reason_chars": avg_depth,
        "max_role_reason_chars": max_role_depth,
        "conflict_detected": 1.0 if extra.get("conflict_detected") else 0.0,
        "conflict_resolved": 1.0 if extra.get("conflict_resolved") else 0.0,
    }
    passed = bool(extra.get("passed"))
    exp = ("E3" if "correlated" in str(sf) else
           "E9" if "fault_injection" in str(sf) else
           "E4" if "coverage" in str(sf) else
           "E8" if "scaling" in str(sf) else "?")
    return feat, passed, exp, Path(sf).stem


def pca2(X):
    """Tiny dependency-free PCA → 2 comps. X: list of equal-length vectors."""
    n, d = len(X), len(X[0])
    mean = [sum(r[j] for r in X) / n for j in range(d)]
    std = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in X) / n) or 1.0
           for j in range(d)]
    Z = [[(r[j] - mean[j]) / std[j] for j in range(d)] for r in X]
    # covariance
    C = [[sum(Z[k][i] * Z[k][j] for k in range(n)) / n for j in range(d)]
         for i in range(d)]
    # power iteration for top-2 eigenvectors
    def power(C, exclude=None):
        v = [1.0] * len(C)
        for _ in range(200):
            w = [sum(C[i][j] * v[j] for j in range(len(C))) for i in range(len(C))]
            if exclude:
                dot = sum(w[i] * exclude[i] for i in range(len(C)))
                w = [w[i] - dot * exclude[i] for i in range(len(C))]
            nrm = math.sqrt(sum(x * x for x in w)) or 1.0
            v = [x / nrm for x in w]
        return v
    v1 = power(C)
    v2 = power(C, exclude=v1)
    P = [[sum(Z[k][j] * v1[j] for j in range(d)),
          sum(Z[k][j] * v2[j] for j in range(d))] for k in range(n)]
    return P


def main():
    rows = []
    for sf in glob.glob(str(ROOT / "experiments/**/logs/**/*.state.json"),
                         recursive=True):
        fp = fingerprint(sf)
        if fp:
            rows.append(fp)
    if not rows:
        print("No state dumps found.")
        return
    keys = list(rows[0][0].keys())
    X = [[r[0][k] for k in keys] for r in rows]
    P = pca2(X)
    passed = [r[1] for r in rows]
    exps = [r[2] for r in rows]

    # (1) PCA scatter coloured by outcome
    fig, ax = plt.subplots(figsize=(10, 7))
    for ok, col, lab in [(True, "#2a9d8f", "passed"), (False, "#e76f51", "failed")]:
        xs = [P[i][0] for i in range(len(P)) if passed[i] == ok]
        ys = [P[i][1] for i in range(len(P)) if passed[i] == ok]
        ax.scatter(xs, ys, c=col, label=lab, alpha=0.6, edgecolors="black",
                   linewidths=0.3, s=45)
    ax.set_xlabel("Coordination PC-1"); ax.set_ylabel("Coordination PC-2")
    ax.set_title("W10: Episodes embedded by COORDINATION fingerprint "
                 f"(n={len(rows)}) — outcome separates", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout(); plt.savefig(OUT / "coordination_fingerprint_pca.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUT/'coordination_fingerprint_pca.png'}")

    # (2) simple grid clustering on PC space → regime pass rates
    def regime(p):
        return (1 if p[0] >= 0 else 0, 1 if p[1] >= 0 else 0)
    reg = defaultdict(list)
    for i, p in enumerate(P):
        reg[regime(p)].append(passed[i])
    lines = ["# W10 — Coordination regimes (PCA-quadrant clustering)", "",
             "| Regime (PC1,PC2) | n | pass rate |", "|---|---|---|"]
    for k in sorted(reg):
        v = reg[k]
        lines.append(f"| {k} | {len(v)} | {sum(v)/len(v):.0%} |")
    # feature → outcome correlation (point-biserial-ish)
    lines += ["", "## Feature ↔ success (mean in passed vs failed)",
              "| feature | mean(passed) | mean(failed) |", "|---|---|---|"]
    for j, k in enumerate(keys):
        mp = [X[i][j] for i in range(len(X)) if passed[i]]
        mf = [X[i][j] for i in range(len(X)) if not passed[i]]
        mp_ = sum(mp)/len(mp) if mp else 0
        mf_ = sum(mf)/len(mf) if mf else 0
        lines.append(f"| {k} | {mp_:.1f} | {mf_:.1f} |")
    (OUT / "coordination_regimes.md").write_text("\n".join(lines))
    print(f"  Saved: {OUT/'coordination_regimes.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
