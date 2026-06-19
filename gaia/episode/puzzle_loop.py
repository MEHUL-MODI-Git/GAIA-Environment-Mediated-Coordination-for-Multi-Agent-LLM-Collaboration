"""Phase-based episode loop for the Asymmetric Information Puzzle experiment.

Architecture (4 explicit phases, all state on shared blackboard):

  Phase 1 — Expert Analysis (parallel)
    Expert-A-1, Expert-A-2 → analyze Partition A → post PLAN (partial_deduction)
    Expert-B-1, Expert-B-2 → analyze Partition B → post PLAN (partial_deduction)

  Phase 2 — Synthesis (parallel)
    Synthesizer-1, Synthesizer-2 → read all deductions → post REVIEW (proposed_solution)

  Phase 3 — Critique
    Critic → compare both solutions → AGREE or post CONFLICT signal
    If CONFLICT → Phase 3b (one re-synthesis round, then continue)

  Phase 4 — Verification
    Verifier → Python constraint solver → Evidence(passed=True/False)

All inter-agent communication happens exclusively via the blackboard.
No direct agent-to-agent calls.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    ArtifactType,
    Policy,
    Signal,
    SignalType,
    Task,
)
from ..agents.puzzle.expert import ExpertAgent
from ..agents.puzzle.synthesizer import SynthesizerAgent
from ..agents.puzzle.puzzle_critic import PuzzleCriticAgent
from ..agents.puzzle.puzzle_verifier import PuzzleVerifierAgent
from ..utils.budget_monitor import BudgetMonitor
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("puzzle_loop")


@dataclass
class PuzzleEpisodeResult:
    puzzle_id: str
    passed: bool
    proposed_solution: Optional[Dict] = None
    ground_truth: Optional[Dict] = None
    num_expert_deductions: int = 0
    num_synthesis_artifacts: int = 0
    conflict_detected: bool = False
    conflict_resolved: bool = False
    iterations: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    stop_reason: str = ""
    error: Optional[str] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)


class PuzzleEpisodeLoop:
    """Orchestrates a single puzzle episode through 4 explicit phases."""

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
            max_cost_per_problem=0.50,
            max_iterations=20,
            max_llm_calls=30,
        )
        self.bb_logger = blackboard.logger

        # Categorize agents by role
        self.expert_a    = [a for a in agents if isinstance(a, ExpertAgent) and a.partition == "A"]
        self.expert_b    = [a for a in agents if isinstance(a, ExpertAgent) and a.partition == "B"]
        self.synthesizers = [a for a in agents if isinstance(a, SynthesizerAgent)]
        self.critics      = [a for a in agents if isinstance(a, PuzzleCriticAgent)]
        self.verifiers    = [a for a in agents if isinstance(a, PuzzleVerifierAgent)]

    async def _dispatch(self, agent, task: Task) -> None:
        """Directly dispatch a task to a specific agent (bypasses polling).

        This avoids the polling priority ordering issue where all tasks have the
        same priority=5.0 and poll_task always returns the first one.
        """
        # Claim the task for this agent
        success = self.blackboard.claim_task(agent.agent_id, task.task_id)
        if not success:
            logger.warning(f"Could not claim task {task.task_id} for {agent.name}")
            return

        self.bb_logger.log_agent_execute(
            agent_id=agent.agent_id, task_id=task.task_id, start=True
        )

        import time
        t0 = time.time()
        try:
            artifacts = await agent.execute(task)
            duration_ms = (time.time() - t0) * 1000
            self.bb_logger.log_agent_execute(
                agent_id=agent.agent_id, task_id=task.task_id,
                start=False, duration_ms=duration_ms
            )
            for artifact in artifacts:
                self.blackboard.post_artifact(artifact)
            self.blackboard.release_task(task.task_id)
        except Exception as e:
            logger.error(f"{agent.name}: dispatch failed: {e}")
            self.blackboard.fail_task(task.task_id, str(e))

    async def run_episode(self, puzzle: Dict[str, Any]) -> PuzzleEpisodeResult:
        """Run complete puzzle episode."""
        puzzle_id = puzzle["puzzle_id"]
        ground_truth = puzzle["solution"]
        clues_a_texts = [c["text"] for c in puzzle["clues_a"]]
        clues_b_texts = [c["text"] for c in puzzle["clues_b"]]
        all_clues_texts = [c["text"] for c in puzzle["all_clues"]]
        all_clues_structs = [c["struct"] for c in puzzle["all_clues"]]
        total_clues = len(puzzle["all_clues"])

        start_time = asyncio.get_event_loop().time()
        self.bb_logger.log_episode_start(problem_id=puzzle_id)
        logger.info(f"=== Puzzle episode: {puzzle_id} ===")

        self.budget_monitor.reset()

        # === Step 1: Post root task ===
        root_task = Task(
            title=f"Solve {puzzle_id}",
            description=f"Logic grid puzzle: {puzzle_id}",
            acceptance_criteria="All 4 people correctly assigned job, pet, and drink",
            metadata={
                "puzzle_id": puzzle_id,
                "solution": ground_truth,
                "all_clues_text": all_clues_texts,
                "all_clues_structs": all_clues_structs,
                "task_type": "puzzle_root",
            },
        )
        self.blackboard.post_task(root_task)

        phase_timings: Dict[str, float] = {}

        # =====================================================================
        # Phase 1: Expert Analysis (all 4 experts run in parallel)
        # =====================================================================
        logger.info("Phase 1: Expert analysis (parallel)...")
        p1_start = asyncio.get_event_loop().time()

        # Mark the root task as claimed by system so it doesn't pollute agent polling
        self.blackboard.claim_task("system", root_task.task_id)

        # Post sub-tasks for each partition (priority 5.0 — highest, returned first by poll)
        expert_a_task = Task(
            parent_id=root_task.task_id,
            title="Analyze Partition A",
            description="Analyze the given clues (Partition A) and post partial deductions",
            priority=5.0,
            metadata={
                "task_type": "expert_a",
                "clues": clues_a_texts,
                "total_clues": total_clues,
                "temperature": 0.2,
            },
        )
        expert_b_task = Task(
            parent_id=root_task.task_id,
            title="Analyze Partition B",
            description="Analyze the given clues (Partition B) and post partial deductions",
            priority=5.0,
            metadata={
                "task_type": "expert_b",
                "clues": clues_b_texts,
                "total_clues": total_clues,
                "temperature": 0.2,
            },
        )
        self.blackboard.post_task(expert_a_task)
        self.blackboard.post_task(expert_b_task)

        # Post a second copy of each expert task (one per parallel agent)
        expert_a_task2 = Task(
            parent_id=root_task.task_id,
            title="Analyze Partition A (agent 2)",
            description="Analyze the given clues (Partition A) and post partial deductions",
            priority=5.0,
            metadata={
                "task_type": "expert_a",
                "clues": clues_a_texts,
                "total_clues": total_clues,
                "temperature": 0.6,
            },
        )
        expert_b_task2 = Task(
            parent_id=root_task.task_id,
            title="Analyze Partition B (agent 2)",
            description="Analyze the given clues (Partition B) and post partial deductions",
            priority=5.0,
            metadata={
                "task_type": "expert_b",
                "clues": clues_b_texts,
                "total_clues": total_clues,
                "temperature": 0.6,
            },
        )
        self.blackboard.post_task(expert_a_task2)
        self.blackboard.post_task(expert_b_task2)

        # Directly dispatch each agent to its specific task (avoids poll ordering issues)
        dispatch_pairs = []
        if len(self.expert_a) >= 1: dispatch_pairs.append((self.expert_a[0], expert_a_task))
        if len(self.expert_a) >= 2: dispatch_pairs.append((self.expert_a[1], expert_a_task2))
        if len(self.expert_b) >= 1: dispatch_pairs.append((self.expert_b[0], expert_b_task))
        if len(self.expert_b) >= 2: dispatch_pairs.append((self.expert_b[1], expert_b_task2))

        await asyncio.gather(*[self._dispatch(agent, task) for agent, task in dispatch_pairs])

        p1_end = asyncio.get_event_loop().time()
        phase_timings["phase1_expert_s"] = round(p1_end - p1_start, 2)

        # Count deductions posted
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        deductions = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
        ]
        num_deductions = len(deductions)
        logger.info(f"Phase 1 complete: {num_deductions} expert deductions posted")
        self.bb_logger.log_iteration(iteration_num=1, start=False)

        # =====================================================================
        # Phase 2: Synthesis (2 synthesizers run in parallel)
        # =====================================================================
        logger.info("Phase 2: Synthesis (parallel)...")
        p2_start = asyncio.get_event_loop().time()

        # Post synthesis tasks (one per synthesizer, each reads all deductions)
        for i, synth in enumerate(self.synthesizers):
            temp = 0.1 if i == 0 else 0.3
            synth_task = Task(
                parent_id=root_task.task_id,
                title=f"Synthesize solution ({synth.name})",
                description="Merge expert deductions into a complete solution",
                priority=4.0,
                metadata={
                    "task_type": "synthesize",
                    "temperature": temp,
                },
            )
            self.blackboard.post_task(synth_task)

        synth_tasks = list(self.blackboard.get_open_tasks(available_only=False))
        synth_tasks = [t for t in synth_tasks if t.metadata.get("task_type") == "synthesize"
                       and t.status.value == "OPEN"]
        dispatch_s = []
        for synth, stask in zip(self.synthesizers, synth_tasks):
            dispatch_s.append((synth, stask))
        await asyncio.gather(*[self._dispatch(agent, task) for agent, task in dispatch_s])

        p2_end = asyncio.get_event_loop().time()
        phase_timings["phase2_synthesis_s"] = round(p2_end - p2_start, 2)

        solution_artifacts = [
            a for a in self.blackboard.get_artifacts_for_task(root_task.task_id)
            if a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
        ]
        logger.info(f"Phase 2 complete: {len(solution_artifacts)} solutions posted")
        self.bb_logger.log_iteration(iteration_num=2, start=False)

        # =====================================================================
        # Phase 3: Critique
        # =====================================================================
        conflict_detected = False
        conflict_resolved = False
        p3_start = asyncio.get_event_loop().time()

        if len(solution_artifacts) >= 2 and self.critics:
            logger.info("Phase 3: Critique...")
            critique_task = Task(
                parent_id=root_task.task_id,
                title="Critique solutions",
                description="Compare the two synthesizer solutions and flag conflicts",
                metadata={"task_type": "critique"},
                priority=2.0,
            )
            self.blackboard.post_task(critique_task)
            await self._dispatch(self.critics[0], critique_task)

            # Check if conflict was raised
            signals = self.blackboard.get_signals(
                signal_type=SignalType.CONFLICT,
                resolved=False,
            )
            conflict_detected = any(s.task_id == root_task.task_id for s in signals)

            if conflict_detected:
                logger.info("Phase 3: CONFLICT detected — triggering re-synthesis (Phase 3b)...")
                self.metrics.record_conflict()

                # Phase 3b: One re-synthesis round
                for synth in self.synthesizers:
                    reconcile_task = Task(
                        parent_id=root_task.task_id,
                        title=f"Reconcile solution ({synth.name})",
                        description=(
                            "Two synthesizers disagreed. Re-read all expert deductions "
                            "and the Critic's analysis, then produce a revised solution."
                        ),
                        metadata={
                            "task_type": "synthesize",
                            "temperature": 0.0,  # Deterministic for reconciliation
                        },
                        priority=3.0,
                    )
                    self.blackboard.post_task(reconcile_task)

                reconcile_tasks = [
                    t for t in self.blackboard.get_open_tasks(available_only=False)
                    if t.metadata.get("task_type") == "synthesize"
                    and t.status.value == "OPEN"
                ]
                recon_pairs = list(zip(self.synthesizers, reconcile_tasks))
                await asyncio.gather(
                    *[self._dispatch(agent, task) for agent, task in recon_pairs]
                )
                conflict_resolved = True
                logger.info("Phase 3b: Re-synthesis complete")

        p3_end = asyncio.get_event_loop().time()
        phase_timings["phase3_critique_s"] = round(p3_end - p3_start, 2)
        self.bb_logger.log_iteration(iteration_num=3, start=False)

        # =====================================================================
        # Phase 4: Verification (Python ground truth)
        # =====================================================================
        logger.info("Phase 4: Verification...")
        p4_start = asyncio.get_event_loop().time()

        verify_task = Task(
            parent_id=root_task.task_id,
            title="Verify solution",
            description="Check the proposed solution against ground truth",
            metadata={
                "task_type": "verify",
                "solution": ground_truth,
                "all_clues_text": all_clues_texts,
                "all_clues_structs": all_clues_structs,
            },
            priority=4.0,
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

        # Read evidence
        all_evidence = []
        final_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        for art in final_artifacts:
            ev_list = self.blackboard.get_evidence_for_artifact(art.artifact_id)
            all_evidence.extend(ev_list)

        # Also check evidence posted directly without artifact_id
        passed = any(e.passed is True for e in all_evidence)
        proposed = None
        for e in all_evidence:
            if e.metadata.get("proposed_solution"):
                proposed = e.metadata["proposed_solution"]
                break

        # Fallback: read from latest REVIEW artifact
        if proposed is None:
            all_arts = self.blackboard.get_artifacts_for_task(root_task.task_id)
            sol_arts = [
                a for a in all_arts
                if a.type == ArtifactType.REVIEW
                and a.metadata.get("subtype") == "proposed_solution"
                and a.metadata.get("parsed_solution")
            ]
            if sol_arts:
                proposed = sol_arts[-1].metadata["parsed_solution"]

        cost = self.budget_monitor.current_cost

        self.bb_logger.log_episode_end(
            problem_id=puzzle_id,
            passed=passed,
            code=str(proposed) if proposed else "",
        )

        logger.info(
            f"Episode done: puzzle={puzzle_id} passed={passed} "
            f"cost=${cost:.4f} duration={duration_s:.1f}s"
        )

        from ..utils.state_dump import auto_dump_episode_state
        auto_dump_episode_state(self.blackboard, puzzle_id, extra={
            "passed": passed, "proposed_solution": proposed,
            "ground_truth": ground_truth,
            "num_expert_deductions": num_deductions,
            "num_synthesis_artifacts": len(solution_artifacts),
            "conflict_detected": conflict_detected,
            "conflict_resolved": conflict_resolved,
            "phase_timings": phase_timings,
        })

        return PuzzleEpisodeResult(
            puzzle_id=puzzle_id,
            passed=passed,
            proposed_solution=proposed,
            ground_truth=ground_truth,
            num_expert_deductions=num_deductions,
            num_synthesis_artifacts=len(solution_artifacts),
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            iterations=3 + (1 if conflict_detected else 0),
            cost_usd=cost,
            duration_s=duration_s,
            stop_reason="passed" if passed else "wrong_answer",
            phase_timings=phase_timings,
        )
