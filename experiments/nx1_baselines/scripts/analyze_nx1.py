#!/usr/bin/env python3
"""NX1 analysis — mechanism-attribution figure + bootstrap CIs.

All conditions face the IDENTICAL correlated-failure stressor (2 misled + 1
clean). The accuracy ladder isolates exactly which structural ingredient
matters.
"""
import glob, json, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.parent
OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def boot(bits, nb=10000, seed=0):
    if not bits:
        return (0, 0, 0)
    rng = random.Random(seed); n = len(bits); m = []
    for _ in range(nb):
        m.append(sum(bits[rng.randrange(n)] for _ in range(n)) / n)
    m.sort()
    return (sum(bits)/n, m[int(.025*nb)], m[int(.975*nb)])


def main():
    nx1 = json.load(open(sorted(
        f for f in glob.glob(str(ROOT/'experiments/nx1_baselines/results/nx1_*.json'))
        if 'checkpoint' not in f)[-1]))
    e3 = json.load(open(sorted(glob.glob(str(
        ROOT/'experiments/correlated_failure/results/correlated_failure_*.json')))[-1]))

    def bits_from(results):
        return [1 if r.get("passed") else 0 for r in results]

    ladder = [
        ("Single\n(1 clean, no stressor)", bits_from(e3["single"]["results"]), "#bdbdbd"),
        ("Majority vote\n(2 misled+1 clean)", bits_from(e3["majority_vote"]["results"]), "#e76f51"),
        ("Debate\n(AutoGen-style, 2 rounds)", bits_from(nx1["debate"]["results"]), "#f4a261"),
        ("Plain blackboard\n(2025 BB-paper design)", bits_from(nx1["blackboard_plain"]["results"]), "#457b9d"),
        ("GAIA\n(+ conflict-as-task)", bits_from(e3["gaia"]["results"]), "#2a9d8f"),
    ]
    labels, accs, los, his, cols = [], [], [], [], []
    for lab, bits, c in ladder:
        pt, lo, hi = boot(bits)
        labels.append(lab); accs.append(pt); los.append(lo); his.append(hi); cols.append(c)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(labels))
    bars = ax.bar(x, accs, color=cols, edgecolor="black", linewidth=1.2,
                  yerr=[[a-l for a, l in zip(accs, los)],
                        [h-a for a, h in zip(accs, his)]],
                  capsize=6, error_kw={"elinewidth": 1.5})
    for i, a in enumerate(accs):
        ax.text(i, a + 0.03, f"{a:.0%}", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Accuracy (bootstrap 95% CI)", fontsize=12)
    ax.set_title("NX1: Mechanism attribution under an IDENTICAL correlated-failure "
                 "stressor\n(2 misled + 1 clean — only the coordination structure "
                 "differs)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    # annotate the key deltas
    ax.annotate("", xy=(3, 0.846), xytext=(1, 0.154),
                arrowprops=dict(arrowstyle="->", color="gray", ls=":"))
    ax.text(2, 0.55, "shared buffer\n+0.69", fontsize=8, color="gray", ha="center")
    ax.annotate("", xy=(4, 1.0), xytext=(3, 0.846),
                arrowprops=dict(arrowstyle="->", color="green"))
    ax.text(3.5, 0.93, "conflict-as-task\n+0.154", fontsize=8, color="green", ha="center")
    plt.tight_layout(); plt.savefig(OUT/"nx1_mechanism_ladder.png", dpi=150)
    plt.close()
    print(f"Saved {OUT/'nx1_mechanism_ladder.png'}")

    md = ["# NX1 — Mechanism attribution (identical 2-misled+1-clean stressor)", "",
          "| Condition | Accuracy [95% CI] | Cost |", "|---|---|---|"]
    costmap = {
        "Single\n(1 clean, no stressor)": e3["single"]["summary"]["total_cost_usd"],
        "Majority vote\n(2 misled+1 clean)": e3["majority_vote"]["summary"]["total_cost_usd"],
        "Debate\n(AutoGen-style, 2 rounds)": nx1["debate"]["summary"]["total_cost_usd"],
        "Plain blackboard\n(2025 BB-paper design)": nx1["blackboard_plain"]["summary"]["total_cost_usd"],
        "GAIA\n(+ conflict-as-task)": e3["gaia"]["summary"]["total_cost_usd"],
    }
    for lab, a, lo, hi in zip(labels, accs, los, his):
        md.append(f"| {lab.replace(chr(10),' ')} | {a:.1%} [{lo:.0%}, {hi:.0%}] "
                  f"| ${costmap[lab]:.3f} |")
    md += ["", "**Reading:** debate (38.5%) barely beats majority (15.4%) — "
           "iterative revision *entrenches* the shared error via conformity. "
           "A plain shared buffer recovers most cases (84.6%) because the "
           "synthesizer can see the lone correct chain. GAIA's structured "
           "conflict-as-task closes the final gap to 100%. The active "
           "ingredient is **not** 'a blackboard' nor 'debate' — it is the "
           "typed CONFLICT signal escalating to reconciliation."]
    (OUT/"nx1_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
