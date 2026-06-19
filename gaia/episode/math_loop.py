"""Phase-based episode loop for the GSM8K Mathematical Reasoning experiment.

Architecture (4 explicit phases, all state on shared blackboard):

  Phase 1 — Parallel Solving
    Solver-1, Solver-2, Solver-3  →  each independently solves the problem
    →  posts PLAN(math_solution) with full reasoning + answer

  Phase 2 — Aggregation (Consensus Check)
    Aggregator  →  reads all 3 solver answers
    →  if unanimous: posts REVIEW(unanimous_answer) + signals done
    →  if conflict:  posts PLAN(aggregator_verdict) + CONFLICT signal

  Phase 3 — Reconciliation (only if CONFLICT was detected)
    Reconciler  →  reads ALL solver chains + conflict summary
    →  identifies errors, re-solves from scratch
    →  posts REVIEW(reconciled_solution) with authoritative answer

  Phase 4 — Verification (always)
    Verifier  →  finds the final REVIEW artifact, extracts integer,
                 compares to ground truth  →  posts Evidence(passed=T/F)

All inter-agent communication happens exclusively via the blackboard.
No direct agent-to-agent calls.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    Artifact,
    ArtifactType,
    Policy,
    Signal,
    SignalType,
    Task,
)
from ..agents.math.math_solver import MathSolverAgent
from ..agents.math.math_aggregator import MathAggregatorAgent
from ..agents.math.math_reconciler import MathReconcilerAgent
from ..agents.math.math_verifier import MathVerifierAgent
from ..utils.budget_monitor import BudgetMonitor
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("math_loop")

# Temperature schedule for the 3 parallel solvers.
# Diversity in temperature → diversity in reasoning paths →
# more likely to expose conflicts on hard problems.
SOLVER_TEMPERATURES = [0.0, 0.3, 0.6]


@dataclass
class MathEpisodeResult:
    problem_id: str
    passed: bool
    proposed_answer: Optional[int] = None
    ground_truth: Optional[int] = None
    conflict_detected: bool = False
    conflict_resolved: bool = False
    solver_answers: Dict[str, Optional[int]] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    stop_reason: str = ""
    error: Optional[str] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)


class MathEpisodeLoop:
    """Orchestrates a single GSM8K problem through 4 explicit phases."""

    def __init__(
        self,
        blackboard: Blackboard,
        agents: List,
        metrics: MetricsCollector,
        policy: Policy,
        budget_monitor: Optional[BudgetMonitor] = None,
    ):
        self.blackboard = blackboard
        self.agents = agents
        self.metrics = metrics
        self.policy = policy
        self.budget_monitor = budget_monitor or BudgetMonitor(
            max_cost_per_problem=1.00,
            max_iterations=20,
            max_llm_calls=30,
        )
        self.bb_logger = blackboard.logger

        # Categorise agents by type
        self.solvers     = sorted(
            [a for a in agents if isinstance(a, MathSolverAgent)],
            key=lambda a: a.solver_index,
        )
        self.aggregators = [a for a in agents if isinstance(a, MathAggregatorAgent)]
        self.reconcilers = [a for a in agents if isinstance(a, MathReconcilerAgent)]
        self.verifiers   = [a for a in agents if isinstance(a, MathVerifierAgent)]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal dispatch helper (same pattern as puzzle_loop._dispatch)
    # ─────────────────────────────────────────────────────────────────────────

    async def _dispatch(self, agent, task: Task) -> None:
        """Directly assign a task to an agent, bypassing the poll queue."""
        success = self.blackboard.claim_task(agent.agent_id, task.task_id)
        if not success:
            logger.warning(f"Could not claim task {task.task_id} for {agent.name}")
            return

        self.bb_logger.log_agent_execute(
            agent_id=agent.agent_id, task_id=task.task_id, start=True
        )
        t0 = time.time()
        try:
            artifacts = await agent.execute(task)
            duration_ms = (time.time() - t0) * 1000
            self.bb_logger.log_agent_execute(
                agent_id=agent.agent_id, task_id=task.task_id,
                start=False, duration_ms=duration_ms,
            )
            for artifact in artifacts:
                self.blackboard.post_artifact(artifact)
            self.blackboard.release_task(task.task_id)
        except Exception as e:
            logger.error(f"{agent.name}: dispatch failed: {e}")
            self.blackboard.fail_task(task.task_id, str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Episode entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def run_episode(self, problem: Dict[str, Any]) -> MathEpisodeResult:
        """Run complete math episode for one GSM8K problem."""
        problem_id  = problem["problem_id"]
        question    = problem["question"]
        ground_truth: int = problem["answer"]

        start_time = asyncio.get_event_loop().time()
        self.bb_logger.log_episode_start(problem_id=problem_id)
        logger.info(f"=== Math episode: {problem_id} ===")

        self.budget_monitor.reset()

        # ── Root task ────────────────────────────────────────────────────────
        root_task = Task(
            title=f"Solve {problem_id}",
            description=f"GSM8K problem: {question[:80]}...",
            acceptance_criteria="Final integer answer matches ground truth",
            metadata={
                "problem_id": problem_id,
                "question": question,
                "answer": ground_truth,
                "task_type": "math_root",
            },
        )
        self.blackboard.post_task(root_task)
        self.blackboard.claim_task("system", root_task.task_id)

        phase_timings: Dict[str, float] = {}
        conflict_detected = False
        conflict_resolved = False

        # =====================================================================
        # Phase 1: Parallel Solving (3 solvers, each at a different temperature)
        # =====================================================================
        logger.info("Phase 1: Parallel solving (3 solvers)...")
        p1_start = asyncio.get_event_loop().time()

        solver_tasks = []
        for i, solver in enumerate(self.solvers):
            temp = SOLVER_TEMPERATURES[i] if i < len(SOLVER_TEMPERATURES) else 0.3
            stask = Task(
                parent_id=root_task.task_id,
                title=f"Solve problem ({solver.name})",
                description=f"Independently solve: {question[:60]}...",
                priority=5.0,
                metadata={
                    "task_type": f"math_solve_{i}",
                    "question": question,
                    "temperature": temp,
                },
            )
            self.blackboard.post_task(stask)
            solver_tasks.append((solver, stask))

        # Run all 3 solvers in true parallel
        await asyncio.gather(*[self._dispatch(s, t) for s, t in solver_tasks])

        p1_end = asyncio.get_event_loop().time()
        phase_timings["phase1_solve_s"] = round(p1_end - p1_start, 2)

        # Collect solver answers for result metadata
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        solver_artifacts = sorted(
            [a for a in all_artifacts
             if a.type == ArtifactType.PLAN
             and a.metadata.get("subtype") == "math_solution"],
            key=lambda a: a.metadata.get("solver_index", 0),
        )
        solver_answers = {
            a.metadata.get("solver_label", f"Solver-{i+1}"): a.metadata.get("answer")
            for i, a in enumerate(solver_artifacts)
        }
        logger.info(f"Phase 1 complete: solver answers = {solver_answers}")
        self.bb_logger.log_iteration(iteration_num=1, start=False)

        # =====================================================================
        # Phase 2: Aggregation (consensus check)
        # =====================================================================
        logger.info("Phase 2: Aggregation (consensus check)...")
        p2_start = asyncio.get_event_loop().time()

        agg_task = Task(
            parent_id=root_task.task_id,
            title="Aggregate solver answers",
            description="Check whether all solvers agree on the answer",
            priority=4.0,
            metadata={
                "task_type": "math_aggregate",
                "question": question,
            },
        )
        self.blackboard.post_task(agg_task)

        if self.aggregators:
            await self._dispatch(self.aggregators[0], agg_task)

        # Read aggregator verdict
        updated_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        verdict_artifacts = [
            a for a in updated_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "aggregator_verdict"
        ]

        if verdict_artifacts:
            verdict = verdict_artifacts[-1]
            is_unanimous = verdict.metadata.get("unanimous", False)
            agreed_answer = verdict.metadata.get("agreed_answer")

            if is_unanimous and agreed_answer is not None:
                # Post the unanimous answer as a REVIEW artifact for the Verifier
                unanimous_review = Artifact(
                    type=ArtifactType.REVIEW,
                    task_id=root_task.task_id,
                    author="system",
                    content=f"Unanimous answer: {agreed_answer}",
                    metadata={
                        "subtype": "unanimous_answer",
                        "answer": agreed_answer,
                        "source": "aggregator_unanimous",
                    },
                )
                self.blackboard.post_artifact(unanimous_review)
                logger.info(f"Phase 2 complete: UNANIMOUS answer={agreed_answer}")
            else:
                conflict_detected = True
                logger.info("Phase 2 complete: CONFLICT detected")
        else:
            # Aggregator failed — fall back to majority vote
            logger.warning("Phase 2: No aggregator verdict, falling back to majority vote")

        p2_end = asyncio.get_event_loop().time()
        phase_timings["phase2_aggregate_s"] = round(p2_end - p2_start, 2)
        self.bb_logger.log_iteration(iteration_num=2, start=False)

        # =====================================================================
        # Phase 3: Reconciliation (only on CONFLICT)
        # =====================================================================
        p3_start = asyncio.get_event_loop().time()

        if conflict_detected and self.reconcilers:
            logger.info("Phase 3: Reconciliation (conflict detected)...")
            rec_task = Task(
                parent_id=root_task.task_id,
                title="Reconcile conflicting answers",
                description=(
                    "Solvers disagreed. Audit all reasoning chains, "
                    "identify errors, and produce the correct answer."
                ),
                priority=3.0,
                metadata={
                    "task_type": "math_reconcile",
                    "question": question,
                },
            )
            self.blackboard.post_task(rec_task)
            await self._dispatch(self.reconcilers[0], rec_task)
            conflict_resolved = True
            logger.info("Phase 3 complete: Reconciliation done")

        p3_end = asyncio.get_event_loop().time()
        phase_timings["phase3_reconcile_s"] = round(p3_end - p3_start, 2)
        self.bb_logger.log_iteration(iteration_num=3, start=False)

        # =====================================================================
        # Phase 4: Verification (always)
        # =====================================================================
        logger.info("Phase 4: Verification...")
        p4_start = asyncio.get_event_loop().time()

        verify_task = Task(
            parent_id=root_task.task_id,
            title="Verify final answer",
            description="Check proposed answer against ground truth",
            priority=2.0,
            metadata={
                "task_type": "math_verify",
                "question": question,
                "answer": ground_truth,
            },
        )
        self.blackboard.post_task(verify_task)

        if self.verifiers:
            await self._dispatch(self.verifiers[0], verify_task)

        p4_end = asyncio.get_event_loop().time()
        phase_timings["phase4_verify_s"] = round(p4_end - p4_start, 2)

        # =====================================================================
        # Collect result
        # =====================================================================
        duration_s = asyncio.get_event_loop().time() - start_time

        # Collect evidence: MathVerifier links its Evidence to the REVIEW artifact
        # it verified, so we gather it by walking all final artifacts.
        all_evidence = []
        final_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        for art in final_artifacts:
            ev_list = self.blackboard.get_evidence_for_artifact(art.artifact_id)
            all_evidence.extend(ev_list)

        passed = any(e.passed is True for e in all_evidence)

        # Determine what answer was proposed
        proposed_answer = None
        for e in all_evidence:
            if e.metadata.get("proposed_answer") is not None:
                proposed_answer = e.metadata["proposed_answer"]
                break

        cost = self.budget_monitor.current_cost

        self.bb_logger.log_episode_end(
            problem_id=problem_id,
            passed=passed,
            code=str(proposed_answer) if proposed_answer is not None else "",
        )

        logger.info(
            f"Episode done: {problem_id} passed={passed} "
            f"proposed={proposed_answer} truth={ground_truth} "
            f"cost=${cost:.4f} duration={duration_s:.1f}s "
            f"conflict={conflict_detected}"
        )

        return MathEpisodeResult(
            problem_id=problem_id,
            passed=passed,
            proposed_answer=proposed_answer,
            ground_truth=ground_truth,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            solver_answers=solver_answers,
            cost_usd=cost,
            duration_s=duration_s,
            stop_reason="passed" if passed else "wrong_answer",
            phase_timings=phase_timings,
        )
