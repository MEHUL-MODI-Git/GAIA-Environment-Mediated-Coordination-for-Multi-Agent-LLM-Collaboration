#!/usr/bin/env python3
"""NX6 — Blackboard visibility-topology sweep.

The communication TOPOLOGY is applied purely as a per-agent READ-FILTER on the
shared blackboard (BlackboardView): the two synthesizers each see only a
topology-restricted subset of the 4 expert deductions. Crucially, SIGNALS are
NOT filtered — so in the GAIA condition the typed CONFLICT signal acts as a
back-channel that re-bridges a sparse topology cut, while the plain-blackboard
condition has no such mechanism.

Topology density k = # of the 4 expert deductions each synthesizer can see:
  k=1  (star-like, severe cut)   k=2 (ring-like)   k=3   k=4 (full mesh)
The two synthesizers are given DIFFERENT subsets so their disagreement is
informative for the GAIA conflict mechanism.

Conditions:
  plain  — experts → 2 synthesizers (topology-restricted) → verify.
           NO critic / NO conflict-as-task. Final = a synthesizer solution.
  gaia   — full PuzzleEpisodeLoop (critic detects synth disagreement →
           conflict-as-task re-synthesis). Same topology restriction on the
           synthesizers' artifact reads; signals pass through.

Hypothesis: GAIA degrades gracefully as k shrinks (conflict signal re-bridges
the cut); plain collapses.

20 puzzles x 4 topologies x 2 conditions.
"""
import asyncio, glob, json, os, random, sys, time, logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
_env = ROOT/".env"
if _env.exists():
    for ln in _env.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.visibility_view import BlackboardView
from gaia.blackboard.models import Policy
from gaia.agents.puzzle import (ExpertAgent, SynthesizerAgent,
                                PuzzleCriticAgent, PuzzleVerifierAgent)
from gaia.agents.puzzle.synthesizer import parse_solution_from_text
from gaia.agents.puzzle.puzzle_verifier import proposed_matches_ground_truth
from gaia.episode.puzzle_loop import PuzzleEpisodeLoop
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor

DATA = ROOT/"data"/"puzzle"/"puzzles.json"
RES = Path(__file__).parent.parent/"results"
LOGS = Path(__file__).parent.parent/"logs"
FAST, SLOW = "gpt-4.1-mini", "gpt-4.1"


def expert_subsets(k, seed):
    """Return (subsetA, subsetB): the expert-name sets each synthesizer sees.
    4 experts: Expert-A-1/A-2/B-1/B-2. k = how many each synth sees; the two
    synthesizers get different (rotated) subsets so disagreement is meaningful."""
    experts = ["Expert-A-1", "Expert-A-2", "Expert-B-1", "Expert-B-2"]
    rng = random.Random(seed)
    order = experts[:]
    rng.shuffle(order)
    s1 = set(order[:k])
    s2 = set(order[-k:])               # overlapping-but-different window
    return s1, s2


async def run_plain(puzzle, k, fast, slow, bud, seed):
    """No critic / no conflict: experts post, 2 topology-restricted
    synthesizers each produce a solution, take the one that parses; verify."""
    pid = puzzle["puzzle_id"]
    bb = Blackboard(log_file=LOGS/f"{pid}_plain_k{k}.jsonl")
    m = MetricsCollector()
    gt = puzzle["solution"]
    ca = [c["text"] for c in puzzle["clues_a"]]
    cb = [c["text"] for c in puzzle["clues_b"]]
    total = len(puzzle["all_clues"])
    experts = [
        ExpertAgent(name="Expert-A-1", partition="A", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-A-2", partition="A", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-B-1", partition="B", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-B-2", partition="B", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
    ]
    # author-id lookup by name
    name2id = {e.name: e.agent_id for e in experts}
    s1, s2 = expert_subsets(k, seed)
    vis1 = {name2id[n] for n in s1}
    vis2 = {name2id[n] for n in s2}

    # post a root task + run experts (write directly, no loop)
    from gaia.blackboard.models import Task, Artifact, ArtifactType
    root = Task(title=f"Solve {pid}", description="puzzle",
                metadata={"puzzle_id": pid, "task_type": "puzzle_root"})
    bb.post_task(root); bb.claim_task("system", root.task_id)

    async def run_expert(e, clues):
        from gaia.prompts.puzzle.expert import ExpertPrompts
        pr = ExpertPrompts()
        msgs = [{"role": "system", "content": pr.SYSTEM},
                {"role": "user", "content": pr.format_user(
                    partition=e.partition, clues=clues, total_clues=total)}]
        resp = await e.call_llm(msgs, temperature=0.2)
        bb.post_artifact(Artifact(type=ArtifactType.PLAN, task_id=root.task_id,
            author=e.agent_id, content=resp,
            metadata={"subtype": "partial_deduction", "partition": e.partition,
                      "agent_name": e.name}))
    await asyncio.gather(
        run_expert(experts[0], ca), run_expert(experts[1], ca),
        run_expert(experts[2], cb), run_expert(experts[3], cb))

    # two topology-restricted synthesizers
    from gaia.prompts.puzzle.synthesizer import SynthesizerPrompts
    sp = SynthesizerPrompts()
    async def synth(view_vis, nm):
        v = BlackboardView(bb, f"plain-{nm}", view_vis)
        arts = [a for a in v.get_artifacts_for_task(root.task_id)
                if a.metadata.get("subtype") == "partial_deduction"]
        ded = [(a.metadata.get("agent_name", a.author),
                a.metadata.get("partition", "?"), a.content) for a in arts]
        if not ded:
            return None
        sa = SynthesizerAgent(name=nm, tier=ModelTier.SLOW, llm=slow,
            blackboard=bb, metrics=m, budget_monitor=bud)
        r = await sa.call_llm([{"role": "system", "content": sp.SYSTEM},
            {"role": "user", "content": sp.format_user(ded)}], temperature=0.1)
        return parse_solution_from_text(r)
    p1, p2 = await asyncio.gather(synth(vis1, "PSynth-1"), synth(vis2, "PSynth-2"))
    proposed = p1 or p2
    ok = bool(proposed and proposed_matches_ground_truth(proposed, gt)[0])
    return ok, bud.current_cost


async def run_gaia(puzzle, k, fast, slow, bud, seed):
    """Full PuzzleEpisodeLoop, but the two synthesizers get topology-restricted
    BlackboardViews (signals NOT filtered → CONFLICT re-bridges the cut)."""
    pid = puzzle["puzzle_id"]
    bb = Blackboard(log_file=LOGS/f"{pid}_gaia_k{k}.jsonl")
    m = MetricsCollector()
    pol = Policy(max_iterations=20, stop_on_first_pass=True)
    experts = [
        ExpertAgent(name="Expert-A-1", partition="A", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-A-2", partition="A", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-B-1", partition="B", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
        ExpertAgent(name="Expert-B-2", partition="B", tier=ModelTier.FAST,
                    llm=fast, blackboard=bb, metrics=m, budget_monitor=bud),
    ]
    name2id = {e.name: e.agent_id for e in experts}
    s1, s2 = expert_subsets(k, seed)
    vis1 = {name2id[n] for n in s1}
    vis2 = {name2id[n] for n in s2}
    syn1 = SynthesizerAgent(name="Synthesizer-1", tier=ModelTier.SLOW, llm=slow,
        blackboard=bb, metrics=m, budget_monitor=bud)
    syn2 = SynthesizerAgent(name="Synthesizer-2", tier=ModelTier.SLOW, llm=slow,
        blackboard=bb, metrics=m, budget_monitor=bud)
    # swap each synthesizer's blackboard for a topology-restricted view
    syn1.blackboard = BlackboardView(bb, syn1.agent_id, vis1)
    syn2.blackboard = BlackboardView(bb, syn2.agent_id, vis2)
    agents = experts + [syn1, syn2,
        PuzzleCriticAgent(name="Critic", tier=ModelTier.FAST, llm=fast,
            blackboard=bb, metrics=m, budget_monitor=bud),
        PuzzleVerifierAgent(name="Verifier", tier=ModelTier.FAST, llm=fast,
            blackboard=bb, metrics=m, budget_monitor=bud)]
    loop = PuzzleEpisodeLoop(blackboard=bb, agents=agents, metrics=m,
                             policy=pol, budget_monitor=bud)
    res = await loop.run_episode(puzzle)
    return bool(res.passed), res.cost_usd


async def main():
    RES.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    puzzles = json.load(open(DATA))["puzzles"]
    fast = OpenAILLM(model=FAST, tier=ModelTier.FAST)
    slow = OpenAILLM(model=SLOW, tier=ModelTier.SLOW)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {}
    EP_TIMEOUT = 180  # hard per-episode wall-clock cap (s) — prevents livelock
    outpath = RES / f"nx6_clean_{ts}.json"
    for cond, fn in [("plain", run_plain), ("gaia", run_gaia)]:
        for k in (1, 2, 3, 4):
            # Visibility floor: gaia k=1 (synth sees only 1/4 experts) is
            # pathological — it livelocked the prior run (two single-view
            # synthesizers can NEVER agree → endless re-synthesis). It is not
            # informative, so we skip it explicitly and record why.
            if cond == "gaia" and k == 1:
                out["gaia_k1"] = {"summary": {"condition": "gaia", "k": 1,
                    "n": 0, "accuracy": None,
                    "skipped": "visibility-floor: gaia k=1 livelocks "
                    "(non-agreeing single-view synthesizers); see NX6 honest "
                    "negative — conflict-as-task needs a visibility floor"}}
                print("gaia  k=1  SKIPPED (visibility floor — livelock cell)")
                continue
            npass = 0; cost = 0.0; recs = []; n_timeout = 0
            for p in puzzles:
                bud = BudgetMonitor(max_cost_per_problem=0.6,
                                    max_iterations=25, max_llm_calls=50)
                try:
                    ok, c = await asyncio.wait_for(
                        fn(p, k, fast, slow, bud,
                           seed=hash(p["puzzle_id"]) & 0xffff),
                        timeout=EP_TIMEOUT)
                except asyncio.TimeoutError:
                    ok, c = False, bud.current_cost; n_timeout += 1
                except Exception:
                    ok, c = False, bud.current_cost
                npass += int(ok); cost += c
                recs.append({"puzzle_id": p["puzzle_id"], "passed": ok})
            acc = npass/len(puzzles)
            out[f"{cond}_k{k}"] = {"summary": {"condition": cond, "k": k,
                "n": len(puzzles), "accuracy": acc, "total_cost_usd": cost,
                "n_timeout": n_timeout},
                "results": recs}
            print(f"{cond:5s} k={k} (synth sees {k}/4 experts)  "
                  f"acc={acc:.0%}  cost=${cost:.3f}  timeouts={n_timeout}")
            json.dump(out, open(outpath, "w"), indent=2, default=str)  # incremental
    json.dump(out, open(outpath, "w"), indent=2, default=str)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
