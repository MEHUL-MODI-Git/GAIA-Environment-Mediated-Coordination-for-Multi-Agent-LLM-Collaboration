"""Episode loop for the Correlated Failure experiment (E3).

Same 4-phase architecture as MathEpisodeLoop, but Phase 1 dispatches a
solver pool engineered to produce a *deterministic correlated failure*:

  - 2 × MisledSolverAgent  — both primed with the SAME plausible-but-wrong
    heuristic for the problem, so they reliably produce the SAME wrong answer.
    This is a controlled instantiation of the "self-consistent error" /
    "correlated failure from shared bias" phenomenon (2025 self-consistency
    literature).
  - 1 × MathSolverAgent (clean) — no hint, produces the correct answer.

The 2 misled solvers form a WRONG 2-vs-1 majority. Then:
  - majority_vote condition → adopts the wrong majority (failure exposed)
  - gaia condition → the Reconciler reads all three reasoning chains (it is
    NEVER given the hint), identifies the flawed shared heuristic, and sides
    with the clean dissenter (the mechanism we test).

Phases 2–4 (Aggregator, Reconciler, Verifier) are unchanged.
"""

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    Artifact, ArtifactType, Policy, Signal, SignalType, Task,
)
from ..agents.math.math_solver import MathSolverAgent
from ..agents.math.misled_solver import MisledSolverAgent
from ..agents.math.math_aggregator import MathAggregatorAgent
from ..agents.math.math_reconciler import MathReconcilerAgent
from ..agents.math.math_verifier import MathVerifierAgent
from ..utils.budget_monitor import BudgetMonitor
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("correlated_failure_loop")


@dataclass
class CorrelatedFailureEpisodeResult:
    problem_id: str
    passed: bool
    proposed_answer: Optional[int] = None
    ground_truth: Optional[int] = None
    misled_answers: Dict[str, Optional[int]] = field(default_factory=dict)
    clean_answer: Optional[int] = None
    majority_answer: Optional[int] = None  # what 2/3 majority vote would say
    correlated_failure_present: bool = False  # 2 misled agree AND wrong
    conflict_detected: bool = False
    conflict_resolved: bool = False
    reconciler_sided_with_clean: bool = False
    cost_usd: float = 0.0
    duration_s: float = 0.0
    stop_reason: str = ""
    error: Optional[str] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)


def _majority_vote(answers: List[Optional[int]]) -> Optional[int]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


class CorrelatedFailureEpisodeLoop:
    """E3 loop: 2 misled solvers + 1 clean solver + aggregator + reconciler."""

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
            max_cost_per_problem=1.00, max_iterations=20, max_llm_calls=30,
        )
        self.bb_logger = blackboard.logger

        all_solvers = [a for a in agents if isinstance(a, MathSolverAgent)]
        self.misled_solvers = sorted(
            [a for a in all_solvers if isinstance(a, MisledSolverAgent)],
            key=lambda a: a.misled_index,
        )
        self.clean_solvers = sorted(
            [a for a in all_solvers if not isinstance(a, MisledSolverAgent)],
            key=lambda a: a.solver_index,
        )
        self.aggregators = [a for a in agents if isinstance(a, MathAggregatorAgent)]
        self.reconcilers = [a for a in agents if isinstance(a, MathReconcilerAgent)]
        self.verifiers = [a for a in agents if isinstance(a, MathVerifierAgent)]

    async def _dispatch(self, agent, task: Task) -> None:
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

    async def run_episode(self, problem: Dict[str, Any]) -> CorrelatedFailureEpisodeResult:
        problem_id = problem["problem_id"]
        question = problem["question"]
        ground_truth: int = problem["answer"]
        hint = problem.get("misleading_hint", "")

        start_time = asyncio.get_event_loop().time()
        self.bb_logger.log_episode_start(problem_id=problem_id)
        logger.info(f"=== Correlated failure episode: {problem_id} ===")

        self.budget_monitor.reset()

        root_task = Task(
            title=f"Solve {problem_id} (correlated failure test)",
            description=f"Trap problem: {question[:80]}...",
            acceptance_criteria="Final integer answer matches ground truth",
            metadata={
                "problem_id": problem_id,
                "question": question,
                "answer": ground_truth,
                "task_type": "math_root",
                "is_trap_problem": True,
                "trap_category": problem.get("category"),
                "common_wrong_answer": problem.get("common_wrong_answer"),
            },
        )
        self.blackboard.post_task(root_task)
        self.blackboard.claim_task("system", root_task.task_id)

        phase_timings: Dict[str, float] = {}
        conflict_detected = False
        conflict_resolved = False

        # ─────────────────────────────────────────────────────────────────
        # Phase 1: 2 misled + 1 clean solver, parallel
        # ─────────────────────────────────────────────────────────────────
        logger.info(
            f"Phase 1: {len(self.misled_solvers)} misled + "
            f"{len(self.clean_solvers)} clean solver(s)"
        )
        p1_start = asyncio.get_event_loop().time()

        dispatch_pairs = []

        for solver in self.misled_solvers:
            mtask = Task(
                parent_id=root_task.task_id,
                title=f"Solve ({solver.name}) — misled",
                description=f"Misled solve: {question[:60]}...",
                priority=5.0,
                metadata={
                    "task_type": f"math_solve_misled_{solver.misled_index}",
                    "question": question,
                    "misleading_hint": hint,
                    "temperature": 0.0,
                },
            )
            self.blackboard.post_task(mtask)
            dispatch_pairs.append((solver, mtask))

        for solver in self.clean_solvers:
            ctask = Task(
                parent_id=root_task.task_id,
                title=f"Solve ({solver.name}) — clean",
                description=f"Clean solve: {question[:60]}...",
                priority=5.0,
                metadata={
                    "task_type": f"math_solve_{solver.solver_index}",
                    "question": question,
                    "temperature": 0.0,
                },
            )
            self.blackboard.post_task(ctask)
            dispatch_pairs.append((solver, ctask))

        await asyncio.gather(*[self._dispatch(a, t) for a, t in dispatch_pairs])

        p1_end = asyncio.get_event_loop().time()
        phase_timings["phase1_solve_s"] = round(p1_end - p1_start, 2)

        all_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        solver_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "math_solution"
        ]
        misled_answers: Dict[str, Optional[int]] = {}
        clean_answer: Optional[int] = None
        for a in solver_artifacts:
            if a.metadata.get("is_misled"):
                misled_answers[a.metadata.get("solver_label", "?")] = a.metadata.get("answer")
            else:
                clean_answer = a.metadata.get("answer")

        all_answers = list(misled_answers.values()) + (
            [clean_answer] if clean_answer is not None else []
        )
        majority_answer = _majority_vote(all_answers)

        # Correlated failure present iff both misled answers agree AND are wrong
        misled_vals = [v for v in misled_answers.values() if v is not None]
        correlated_failure = (
            len(misled_vals) >= 2
            and len(set(misled_vals)) == 1
            and misled_vals[0] != ground_truth
        )

        logger.info(
            f"Phase 1: misled={misled_answers} clean={clean_answer} "
            f"majority={majority_answer} correlated_failure={correlated_failure}"
        )
        self.bb_logger.log_iteration(iteration_num=1, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 2: Aggregation (consensus check)
        # ─────────────────────────────────────────────────────────────────
        logger.info("Phase 2: Aggregation...")
        p2_start = asyncio.get_event_loop().time()

        agg_task = Task(
            parent_id=root_task.task_id,
            title="Aggregate solver answers",
            description="Check whether all solvers agree on the answer",
            priority=4.0,
            metadata={"task_type": "math_aggregate", "question": question},
        )
        self.blackboard.post_task(agg_task)

        if self.aggregators:
            await self._dispatch(self.aggregators[0], agg_task)

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
                logger.info(f"Phase 2: UNANIMOUS answer={agreed_answer}")
            else:
                conflict_detected = True
                logger.info("Phase 2: CONFLICT detected")

        p2_end = asyncio.get_event_loop().time()
        phase_timings["phase2_aggregate_s"] = round(p2_end - p2_start, 2)
        self.bb_logger.log_iteration(iteration_num=2, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 3: Reconciliation (on conflict)
        # ─────────────────────────────────────────────────────────────────
        p3_start = asyncio.get_event_loop().time()

        if conflict_detected and self.reconcilers:
            logger.info("Phase 3: Reconciliation (audit + correct)...")
            rec_task = Task(
                parent_id=root_task.task_id,
                title="Reconcile conflicting answers",
                description=(
                    "Solvers disagreed. Audit ALL reasoning chains, identify "
                    "any flawed shared assumption or heuristic, and produce the "
                    "correct answer from first principles."
                ),
                priority=3.0,
                metadata={"task_type": "math_reconcile", "question": question},
            )
            self.blackboard.post_task(rec_task)
            await self._dispatch(self.reconcilers[0], rec_task)
            conflict_resolved = True

        p3_end = asyncio.get_event_loop().time()
        phase_timings["phase3_reconcile_s"] = round(p3_end - p3_start, 2)
        self.bb_logger.log_iteration(iteration_num=3, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 4: Verification
        # ─────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────
        # Collect result
        # ─────────────────────────────────────────────────────────────────
        duration_s = asyncio.get_event_loop().time() - start_time

        all_evidence = []
        final_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        for art in final_artifacts:
            ev_list = self.blackboard.get_evidence_for_artifact(art.artifact_id)
            all_evidence.extend(ev_list)

        passed = any(e.passed is True for e in all_evidence)
        proposed_answer = None
        for e in all_evidence:
            if e.metadata.get("proposed_answer") is not None:
                proposed_answer = e.metadata["proposed_answer"]
                break

        reconciler_sided_with_clean = (
            conflict_resolved
            and proposed_answer is not None
            and clean_answer is not None
            and proposed_answer == clean_answer
        )

        cost = self.budget_monitor.current_cost

        self.bb_logger.log_episode_end(
            problem_id=problem_id,
            passed=passed,
            code=str(proposed_answer) if proposed_answer is not None else "",
        )

        logger.info(
            f"Episode done: {problem_id} passed={passed} "
            f"proposed={proposed_answer} truth={ground_truth} "
            f"clean={clean_answer} majority={majority_answer} "
            f"reconciler_sided_with_clean={reconciler_sided_with_clean}"
        )

        from ..utils.state_dump import auto_dump_episode_state
        auto_dump_episode_state(self.blackboard, problem_id, extra={
            "passed": passed, "proposed_answer": proposed_answer,
            "ground_truth": ground_truth, "clean_answer": clean_answer,
            "misled_answers": misled_answers, "majority_answer": majority_answer,
            "correlated_failure_present": correlated_failure,
            "conflict_detected": conflict_detected,
            "conflict_resolved": conflict_resolved,
            "reconciler_sided_with_clean": reconciler_sided_with_clean,
            "phase_timings": phase_timings,
        })

        return CorrelatedFailureEpisodeResult(
            problem_id=problem_id,
            passed=passed,
            proposed_answer=proposed_answer,
            ground_truth=ground_truth,
            misled_answers=misled_answers,
            clean_answer=clean_answer,
            majority_answer=majority_answer,
            correlated_failure_present=correlated_failure,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            reconciler_sided_with_clean=reconciler_sided_with_clean,
            cost_usd=cost,
            duration_s=duration_s,
            stop_reason="passed" if passed else "wrong_answer",
            phase_timings=phase_timings,
        )
