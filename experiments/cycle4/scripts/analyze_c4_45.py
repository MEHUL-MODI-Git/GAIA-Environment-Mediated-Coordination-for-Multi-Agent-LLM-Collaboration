#!/usr/bin/env python3
"""C4-4/C4-5 figure — causal attribution (leave-one-out) + do-operator.

Two panels on the 13 E3 traps (gpt-4.1-nano, same model as C4-1):
  LEFT  C4-4 leave-one-agent-out: accuracy with each role ablated. Isolates
        the TRIGGER — which role's removal kills conflict-as-task.
  RIGHT C4-5 counterfactual do-operator: reconciler replayed on surgically
        edited transcripts. Isolates the reconciler's INTRINSIC reasoning
        once it is invoked.

Honest synthesis (printed + in the caption): the two panels look like they
disagree (drop-clean = 0% vs do(clean:=trap) = 100%) but they don't — they
measure different things. See module docstring tail.
"""
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.parent
RES = ROOT/"experiments"/"cycle4"/"results"
FIG = ROOT/"experiments"/"cycle4"/"figures"
BLUE, GREEN, RED, GREY = "#1565C0", "#2E7D32", "#B71C1C", "#455A64"


def latest(pat):
    f = sorted(glob.glob(str(RES/pat)))
    if not f:
        raise SystemExit(f"missing {pat}")
    return json.load(open(f[-1]))


def main():
    d4 = latest("c4_4_openai_*.json")
    d5 = latest("c4_5_openai_*.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

    order4 = ["full", "-misled0", "-misled1", "-clean", "-reconciler"]
    lab4 = ["full\npipeline", "−misled0\n(redundant)", "−misled1\n(redundant)",
            "−clean\n(DISSENTER)", "−reconciler\n(→ majority vote)"]
    v4 = [d4[k]["accuracy"]*100 for k in order4]
    c4 = [BLUE, GREEN, GREEN, RED, RED]
    b1 = ax1.bar(range(5), v4, color=c4, edgecolor="k", lw=0.6)
    ax1.bar_label(b1, labels=[f"{x:.0f}%" for x in v4], padding=3, fontsize=10)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(lab4, fontsize=8.6)
    ax1.set_ylabel("Accuracy on 13 E3 traps (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("C4-4  Leave-one-agent-out — which role is load-bearing")
    ax1.text(0.0, -0.31, "Redundant misled solver: Δ 0.\nClean DISSENTER or "
             "reconciler: Δ −100\n(no dissent ⇒ no CONFLICT ⇒\nconflict-as-task "
             "never fires).",
             transform=ax1.transAxes, ha="left", va="top",
             fontsize=8.4, color=GREY)

    order5 = ["baseline", "do_clean=trap", "do_misled0=correct",
              "do_clean=empty"]
    lab5 = ["baseline\n(2 misled+1 clean)", "do(clean:=trap)\nALL 3 wrong",
            "do(misled0:=correct)\n2 correct+1 misled", "do(clean:=empty)\ndissent blanked"]
    v5 = [d5[k]["accuracy"]*100 for k in order5]
    c5 = [BLUE, RED, GREEN, "#E65100"]
    b2 = ax2.bar(range(4), v5, color=c5, edgecolor="k", lw=0.6)
    ax2.bar_label(b2, labels=[f"{x:.0f}%" for x in v5], padding=3, fontsize=10)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(lab5, fontsize=8.4)
    ax2.set_ylabel("Reconciler recovers truth (%)")
    ax2.set_ylim(0, 112)
    ax2.set_title("C4-5  Counterfactual do-operator — reconciler, once invoked")
    ax2.text(1.0, -0.31, "Forced to run on a unanimous-WRONG board: still "
             "100%.\nTracks a corrected majority: 100%.\nNeeds dissent's "
             "PRESENCE not content (−8% blanked).\nNot a vote-follower — an "
             "independent re-deriver.",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=8.4, color=GREY)

    fig.suptitle("GAIA causal decomposition: the dissenter is the TRIGGER, "
                 "the reconciler is the independent RE-DERIVER "
                 "(gpt-4.1-nano, 13 E3 traps)", fontsize=11.5, y=1.02)
    fig.subplots_adjust(bottom=0.30, wspace=0.22)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/"c4_45_causal_decomposition.png", dpi=180,
                bbox_inches="tight")
    print(f"saved {FIG/'c4_45_causal_decomposition.png'}")
    print("C4-4:", {k: d4[k]["accuracy"] for k in order4})
    print("C4-5:", {k: d5[k]["accuracy"] for k in order5})
    print("\nSynthesis: −clean=0% (no 3rd answer ⇒ no CONFLICT ⇒ pipeline "
          "never escalates) vs do(clean:=trap)=100% (reconciler FORCED to "
          "run anyway still re-derives truth). Different objects: C4-4 = the "
          "escalation TRIGGER; C4-5 = the reconciler's intrinsic capability. "
          "Together: dissent creates the conflict; the reconciler, once "
          "triggered, recovers truth by independent reasoning, not voting.")


if __name__ == "__main__":
    main()
