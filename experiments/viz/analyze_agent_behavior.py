#!/usr/bin/env python3
"""Cross-experiment agent-behaviour & pipeline-state analyzer.

Mines the per-episode `.state.json` dumps (full blackboard: tasks, artifacts
WITH reasoning content, signals, evidence) plus the `.jsonl` event logs to
produce a structured behavioural dataset for the paper:

  Per agent role:
    - # artifacts produced, mean reasoning-chain length (chars)
    - role in the pipeline (solver / synthesizer / auditor / reconciler / ...)

  Per episode:
    - pipeline phase timings
    - signal timeline (CONFLICT/UNCERTAINTY/... with descriptions)
    - conflict → resolution outcome

  Reasoning-chain extracts (qualitative, for paper appendix):
    - For E3: the flawed heuristic the misled solvers applied, and the
      reconciler's diagnosis text.
    - For E9: the auditor's trust-score reasoning + contradiction list.

Outputs:
  <exp>/figures/behavior/agent_roles.png       — artifacts & reasoning length per role
  <exp>/figures/behavior/signal_timeline.png   — signals per episode
  <exp>/figures/behavior/behavior_dataset.json — full structured dump
  <exp>/figures/behavior/reasoning_samples.md  — qualitative chain extracts

Usage:
  python experiments/viz/analyze_agent_behavior.py --exp correlated_failure
  python experiments/viz/analyze_agent_behavior.py --exp fault_injection
  python experiments/viz/analyze_agent_behavior.py --exp puzzle --subdir coverage
"""

import argparse
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXP_DIRS = {
    "correlated_failure": PROJECT_ROOT / "experiments" / "correlated_failure",
    "fault_injection":    PROJECT_ROOT / "experiments" / "fault_injection",
    "puzzle":             PROJECT_ROOT / "experiments" / "puzzle",
    "humaneval":          PROJECT_ROOT / "experiments" / "humaneval",
}


def find_state_files(exp_dir: Path, subdir: str = ""):
    logs = exp_dir / "logs"
    if subdir:
        logs = logs / subdir
    return sorted(logs.rglob("*.state.json"))


def role_from_author(state, author):
    """Infer an agent's role from the artifacts/metadata it produced."""
    for a in state["artifacts"].values():
        if a["author"] == author:
            md = a.get("metadata", {})
            if md.get("is_misled"):
                return "misled_solver"
            if md.get("subtype") == "math_solution":
                return "clean_solver"
            if md.get("subtype") == "reconciled_solution":
                return "reconciler"
            if md.get("subtype") == "aggregator_verdict":
                return "aggregator"
            if md.get("subtype") == "partial_deduction":
                return "faulty_expert" if md.get("is_faulty") else "expert"
            if md.get("subtype") == "proposed_solution":
                return "synthesizer"
            if md.get("subtype") == "trust_audit":
                return "deduction_auditor"
            if md.get("subtype") == "generalist_solution":
                return "generalist"
            agent_name = md.get("agent_name", "")
            if agent_name:
                return agent_name.split("-")[0].lower()
    return "unknown"


def analyze(state_files):
    role_artifacts = defaultdict(int)
    role_reasoning_len = defaultdict(list)
    signal_counts = Counter()
    episode_signals = {}
    phase_timings = defaultdict(list)
    episodes = []

    for sf in state_files:
        try:
            st = json.load(open(sf))
        except Exception:
            continue
        eid = st.get("episode_id", sf.stem)
        extra = st.get("extra", {})
        episodes.append({"episode_id": eid, "file": str(sf), **{
            k: extra.get(k) for k in
            ("passed", "conflict_detected", "conflict_resolved",
             "reconciler_sided_with_clean", "auditor_flagged_faulty_agent",
             "correlated_failure_present")
        }})

        for a in st["artifacts"].values():
            role = role_from_author(st, a["author"])
            role_artifacts[role] += 1
            role_reasoning_len[role].append(len(a.get("content", "") or ""))

        sigs = []
        for s in st["signals"].values():
            signal_counts[s["type"]] += 1
            sigs.append({"type": s["type"], "desc": (s.get("description") or "")[:120]})
        episode_signals[eid] = sigs

        for k, v in (extra.get("phase_timings") or {}).items():
            phase_timings[k].append(v)

    return {
        "role_artifacts": dict(role_artifacts),
        "role_reasoning_len_mean": {
            r: round(statistics.mean(v), 1) for r, v in role_reasoning_len.items() if v
        },
        "signal_counts": dict(signal_counts),
        "episode_signals": episode_signals,
        "phase_timing_mean_s": {
            k: round(statistics.mean(v), 2) for k, v in phase_timings.items() if v
        },
        "n_episodes": len(episodes),
        "episodes": episodes,
    }


def plot_agent_roles(data, out_path):
    roles = sorted(data["role_artifacts"], key=lambda r: -data["role_artifacts"][r])
    if not roles:
        return
    counts = [data["role_artifacts"][r] for r in roles]
    rlen = [data["role_reasoning_len_mean"].get(r, 0) for r in roles]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(roles))
    ax1.bar(x, counts, color="#2a9d8f", alpha=0.8, label="# artifacts")
    ax1.set_ylabel("# artifacts produced", color="#2a9d8f")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(roles, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, rlen, "o-", color="#e76f51", linewidth=2, label="mean reasoning chars")
    ax2.set_ylabel("mean reasoning-chain length (chars)", color="#e76f51")
    ax1.set_title("Agent Behaviour by Role", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_signal_timeline(data, out_path):
    if not data["signal_counts"]:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    types = list(data["signal_counts"])
    vals = [data["signal_counts"][t] for t in types]
    ax.bar(types, vals, color="#d32f2f", edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("total occurrences")
    ax.set_title(f"Signals across {data['n_episodes']} episodes",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def extract_reasoning_samples(state_files, out_path, max_samples=3):
    """Pull qualitative reasoning chains for the paper appendix."""
    lines = ["# Reasoning-Chain Samples (for paper appendix)\n"]
    n = 0
    for sf in state_files:
        if n >= max_samples:
            break
        try:
            st = json.load(open(sf))
        except Exception:
            continue
        arts = list(st["artifacts"].values())
        misled = [a for a in arts if a.get("metadata", {}).get("is_misled")]
        recon = [a for a in arts if a.get("metadata", {}).get("subtype") == "reconciled_solution"]
        audit = [a for a in arts if a.get("metadata", {}).get("subtype") == "trust_audit"]
        if not (misled or recon or audit):
            continue
        n += 1
        lines.append(f"\n## Episode: {st.get('episode_id')}\n")
        if misled:
            lines.append("### Misled solver reasoning (flawed shared heuristic)\n")
            lines.append("```\n" + (misled[0].get("content", "")[:1200]) + "\n```\n")
        if recon:
            lines.append("### Reconciler diagnosis (overrides wrong majority)\n")
            lines.append("```\n" + (recon[0].get("content", "")[:1500]) + "\n```\n")
        if audit:
            lines.append("### Deduction-auditor trust analysis\n")
            lines.append("```\n" + (audit[0].get("content", "")[:1500]) + "\n```\n")
    out_path.write_text("\n".join(lines))
    print(f"  Saved: {out_path} ({n} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXP_DIRS))
    parser.add_argument("--subdir", default="", help="logs subdir (e.g. coverage, scaling)")
    args = parser.parse_args()

    exp_dir = EXP_DIRS[args.exp]
    state_files = find_state_files(exp_dir, args.subdir)
    if not state_files:
        print(f"No .state.json dumps found under {exp_dir}/logs"
              f"{'/' + args.subdir if args.subdir else ''}")
        print("(state dumps are produced by episode loops on the gaia/full-pipeline conditions)")
        return

    print(f"Analyzing {len(state_files)} state dumps for '{args.exp}'...")
    data = analyze(state_files)

    # Subdir-aware output so multi-variant experiments (puzzle: coverage vs
    # scaling) do not overwrite each other's behaviour datasets.
    out_dir = exp_dir / "figures" / (args.subdir or "") / "behavior" \
        if args.subdir else exp_dir / "figures" / "behavior"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_agent_roles(data, out_dir / "agent_roles.png")
    plot_signal_timeline(data, out_dir / "signal_timeline.png")
    extract_reasoning_samples(state_files, out_dir / "reasoning_samples.md")

    with open(out_dir / "behavior_dataset.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {out_dir / 'behavior_dataset.json'}")

    print("\n── Behaviour summary ──")
    print(f"  episodes analyzed : {data['n_episodes']}")
    print(f"  artifacts by role : {data['role_artifacts']}")
    print(f"  mean reasoning len: {data['role_reasoning_len_mean']}")
    print(f"  signal counts     : {data['signal_counts']}")
    print(f"  phase timings (s) : {data['phase_timing_mean_s']}")


if __name__ == "__main__":
    main()
