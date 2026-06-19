"""Episode loop for the Fault Injection experiment (E9).

Extends the standard puzzle pipeline with two additional phases:

  Phase 1: Expert Analysis (parallel)
    Standard ExpertAgents + (in fault conditions) one FaultyExpertAgent that
    receives a mix of correct and corrupted clues.

  Phase 1b: Deduction Audit (NEW)
    DeductionAuditorAgent reads all partial deductions, cross-checks for
    logical contradictions, posts a trust score table + (if contradictions
    found) a CONFLICT signal.

  Phase 2: Trust-Aware Synthesis
    TrustAwareSynthesizerAgent reads the trust audit and synthesizes a
    solution that weights low-trust deductions accordingly.

  Phase 3: Critique (standard)

  Phase 4: Verification (standard)

The key contrast tested by E9:
  clean_gaia       — no fault, no auditor — accuracy is baseline GAIA accuracy
  fault_standard   — fault injected, NO auditor — accuracy should drop
  fault_gaia       — fault injected + auditor + trust-aware synth — accuracy recovers
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    Artifact, ArtifactType, Policy, Signal, SignalType, Task,
)
from ..agents.puzzle.expert import ExpertAgent
from ..agents.puzzle.faulty_expert import FaultyExpertAgent
from ..agents.puzzle.synthesizer import SynthesizerAgent
from ..agents.puzzle.deduction_auditor import DeductionAuditorAgent
from ..agents.puzzle.trust_synthesizer import TrustAwareSynthesizerAgent
from ..agents.puzzle.puzzle_critic import PuzzleCriticAgent
from ..agents.puzzle.puzzle_verifier import PuzzleVerifierAgent
from ..utils.budget_monitor import BudgetMonitor
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("fault_injection_loop")


@dataclass
class FaultInjectionEpisodeResult:
    puzzle_id: str
    passed: bool
    proposed_solution: Optional[Dict] = None
    ground_truth: Optional[Dict] = None
    num_expert_deductions: int = 0
    num_synthesis_artifacts: int = 0
    fault_injected: bool = False
    auditor_used: bool = False
    auditor_flagged_faulty_agent: bool = False
    auditor_suspect_id: Optional[str] = None
    real_faulty_agent_id: Optional[str] = None
    trust_scores: Dict[str, float] = field(default_factory=dict)
    n_contradictions_found: int = 0
    conflict_detected: bool = False
    conflict_resolved: bool = False
    iterations: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    stop_reason: str = ""
    error: Optional[str] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)


class FaultInjectionEpisodeLoop:
    """Puzzle loop with optional fault injection + auditor + trust-aware synth."""

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
            max_cost_per_problem=1.00, max_iterations=20, max_llm_calls=40,
        )
        self.bb_logger = blackboard.logger

        # Categorise agents
        all_experts = [a for a in agents if isinstance(a, ExpertAgent)]
        self.expert_a = [a for a in all_experts if a.partition == "A"]
        self.expert_b = [a for a in all_experts if a.partition == "B"]
        self.faulty_experts = [a for a in agents if isinstance(a, FaultyExpertAgent)]
        # Note: TrustAwareSynthesizerAgent IS-A SynthesizerAgent
        self.synthesizers = [a for a in agents if isinstance(a, SynthesizerAgent)]
        self.auditors = [a for a in agents if isinstance(a, DeductionAuditorAgent)]
        self.critics = [a for a in agents if isinstance(a, PuzzleCriticAgent)]
        self.verifiers = [a for a in agents if isinstance(a, PuzzleVerifierAgent)]

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

    async def run_episode(self, puzzle: Dict[str, Any]) -> FaultInjectionEpisodeResult:
        puzzle_id = puzzle["puzzle_id"]
        ground_truth = puzzle["solution"]
        clues_a_texts = [c["text"] for c in puzzle["clues_a"]]
        clues_b_texts = [c["text"] for c in puzzle["clues_b"]]
        all_clues_texts = [c["text"] for c in puzzle["all_clues"]]
        all_clues_structs = [c["struct"] for c in puzzle["all_clues"]]
        total_clues = len(puzzle["all_clues"])

        start_time = asyncio.get_event_loop().time()
        self.bb_logger.log_episode_start(problem_id=puzzle_id)
        logger.info(f"=== Fault injection episode: {puzzle_id} ===")

        self.budget_monitor.reset()

        # Root task
        root_task = Task(
            title=f"Solve {puzzle_id} (fault-injection)",
            description=f"Logic puzzle with possible fault injection: {puzzle_id}",
            acceptance_criteria="All people correctly assigned all attributes",
            metadata={
                "puzzle_id": puzzle_id,
                "solution": ground_truth,
                "all_clues_text": all_clues_texts,
                "all_clues_structs": all_clues_structs,
                "task_type": "puzzle_root",
                "fault_injected": len(self.faulty_experts) > 0,
                "auditor_used": len(self.auditors) > 0,
            },
        )
        self.blackboard.post_task(root_task)
        self.blackboard.claim_task("system", root_task.task_id)

        phase_timings: Dict[str, float] = {}

        # ─────────────────────────────────────────────────────────────────
        # Phase 1: Expert Analysis (including any FaultyExperts)
        # ─────────────────────────────────────────────────────────────────
        logger.info("Phase 1: Expert analysis (with potential fault injection)")
        p1_start = asyncio.get_event_loop().time()

        # Build dispatch pairs. All experts (real or faulty) claim the
        # expert_a / expert_b task type, so we just give each expert agent
        # its own task. FaultyExpertAgent inherits from ExpertAgent so it's
        # in self.expert_a or self.expert_b by partition.
        # But wait — FaultyExpertAgent should also be counted by partition.
        # Let's re-categorize: get ALL expert-like agents grouped by partition.
        all_experts_a = self.expert_a + [a for a in self.faulty_experts if a.partition == "A"]
        all_experts_b = self.expert_b + [a for a in self.faulty_experts if a.partition == "B"]

        # Deduplicate (in case isinstance picked up faulty experts in self.expert_a)
        seen = set()
        all_experts_a = [a for a in all_experts_a if not (a.agent_id in seen or seen.add(a.agent_id))]
        seen = set()
        all_experts_b = [a for a in all_experts_b if not (a.agent_id in seen or seen.add(a.agent_id))]

        dispatch_pairs = []
        for i, agent in enumerate(all_experts_a):
            temp = 0.2 + 0.2 * (i % 3)  # 0.2, 0.4, 0.6 cycle
            task = Task(
                parent_id=root_task.task_id,
                title=f"Analyze Partition A ({agent.name})",
                description="Analyze the given clues (Partition A) and post partial deductions",
                priority=5.0,
                metadata={
                    "task_type": "expert_a",
                    "clues": clues_a_texts,
                    "total_clues": total_clues,
                    "temperature": temp,
                },
            )
            self.blackboard.post_task(task)
            dispatch_pairs.append((agent, task))

        for i, agent in enumerate(all_experts_b):
            temp = 0.2 + 0.2 * (i % 3)
            task = Task(
                parent_id=root_task.task_id,
                title=f"Analyze Partition B ({agent.name})",
                description="Analyze the given clues (Partition B) and post partial deductions",
                priority=5.0,
                metadata={
                    "task_type": "expert_b",
                    "clues": clues_b_texts,
                    "total_clues": total_clues,
                    "temperature": temp,
                },
            )
            self.blackboard.post_task(task)
            dispatch_pairs.append((agent, task))

        await asyncio.gather(*[self._dispatch(a, t) for a, t in dispatch_pairs])

        p1_end = asyncio.get_event_loop().time()
        phase_timings["phase1_expert_s"] = round(p1_end - p1_start, 2)

        # Collect deduction count
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task.task_id)
        deductions = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
        ]
        num_deductions = len(deductions)
        logger.info(f"Phase 1 complete: {num_deductions} expert deductions posted")
        self.bb_logger.log_iteration(iteration_num=1, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 1b: Deduction Audit (NEW — only if auditor present)
        # ─────────────────────────────────────────────────────────────────
        auditor_flagged_faulty = False
        auditor_suspect_id = None
        trust_scores: Dict[str, float] = {}
        n_contradictions = 0

        if self.auditors:
            logger.info("Phase 1b: Deduction audit...")
            p1b_start = asyncio.get_event_loop().time()

            audit_task = Task(
                parent_id=root_task.task_id,
                title="Audit expert deductions",
                description="Cross-check all expert deductions for logical contradictions",
                priority=4.5,
                metadata={"task_type": "deduction_audit"},
            )
            self.blackboard.post_task(audit_task)
            await self._dispatch(self.auditors[0], audit_task)

            # Read the audit result
            audit_arts = [
                a for a in self.blackboard.get_artifacts_for_task(root_task.task_id)
                if a.type == ArtifactType.DOCUMENTATION
                and a.metadata.get("subtype") == "trust_audit"
            ]
            if audit_arts:
                latest = sorted(audit_arts, key=lambda a: a.created_at)[-1]
                trust_scores = latest.metadata.get("trust_scores", {})
                auditor_suspect_id = latest.metadata.get("suspected_faulty_agent_id")
                n_contradictions = len(latest.metadata.get("contradictions", []))
                if auditor_suspect_id and self.faulty_experts:
                    real_id = self.faulty_experts[0].agent_id
                    auditor_flagged_faulty = (auditor_suspect_id == real_id)
                logger.info(
                    f"Auditor: {n_contradictions} contradictions, "
                    f"suspect={auditor_suspect_id[:8] if auditor_suspect_id else 'NONE'}, "
                    f"correctly_flagged={auditor_flagged_faulty}"
                )

            p1b_end = asyncio.get_event_loop().time()
            phase_timings["phase1b_audit_s"] = round(p1b_end - p1b_start, 2)

        # ─────────────────────────────────────────────────────────────────
        # Phase 2: Synthesis (trust-aware if auditor was used)
        # ─────────────────────────────────────────────────────────────────
        logger.info("Phase 2: Synthesis (parallel)...")
        p2_start = asyncio.get_event_loop().time()

        for i, synth in enumerate(self.synthesizers):
            temp = 0.1 if i == 0 else 0.3
            stask = Task(
                parent_id=root_task.task_id,
                title=f"Synthesize solution ({synth.name})",
                description="Merge expert deductions into a complete solution",
                priority=4.0,
                metadata={"task_type": "synthesize", "temperature": temp},
            )
            self.blackboard.post_task(stask)

        synth_tasks = [
            t for t in self.blackboard.get_open_tasks(available_only=False)
            if t.metadata.get("task_type") == "synthesize" and t.status.value == "OPEN"
        ]
        dispatch_s = list(zip(self.synthesizers, synth_tasks))
        await asyncio.gather(*[self._dispatch(a, t) for a, t in dispatch_s])

        p2_end = asyncio.get_event_loop().time()
        phase_timings["phase2_synthesis_s"] = round(p2_end - p2_start, 2)

        solution_artifacts = [
            a for a in self.blackboard.get_artifacts_for_task(root_task.task_id)
            if a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
        ]
        logger.info(f"Phase 2 complete: {len(solution_artifacts)} solutions posted")
        self.bb_logger.log_iteration(iteration_num=2, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 3: Critique
        # ─────────────────────────────────────────────────────────────────
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

            signals = self.blackboard.get_signals(
                signal_type=SignalType.CONFLICT,
                resolved=False,
            )
            # Filter out the auditor's CONFLICT signal (which is about deductions,
            # not synthesizer disagreement)
            synth_conflict = any(
                s.task_id == root_task.task_id
                and s.metadata.get("source") != "deduction_auditor"
                for s in signals
            )
            conflict_detected = synth_conflict

            if conflict_detected:
                logger.info("Phase 3: Synth CONFLICT — re-synthesizing")
                self.metrics.record_conflict()
                for synth in self.synthesizers:
                    reconcile_task = Task(
                        parent_id=root_task.task_id,
                        title=f"Reconcile solution ({synth.name})",
                        description="Synthesizers disagreed; produce revised solution.",
                        metadata={"task_type": "synthesize", "temperature": 0.0},
                        priority=3.0,
                    )
                    self.blackboard.post_task(reconcile_task)
                reconcile_tasks = [
                    t for t in self.blackboard.get_open_tasks(available_only=False)
                    if t.metadata.get("task_type") == "synthesize"
                    and t.status.value == "OPEN"
                ]
                recon_pairs = list(zip(self.synthesizers, reconcile_tasks))
                await asyncio.gather(*[self._dispatch(a, t) for a, t in recon_pairs])
                conflict_resolved = True

        p3_end = asyncio.get_event_loop().time()
        phase_timings["phase3_critique_s"] = round(p3_end - p3_start, 2)
        self.bb_logger.log_iteration(iteration_num=3, start=False)

        # ─────────────────────────────────────────────────────────────────
        # Phase 4: Verification
        # ─────────────────────────────────────────────────────────────────
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
        proposed = None
        for e in all_evidence:
            if e.metadata.get("proposed_solution"):
                proposed = e.metadata["proposed_solution"]
                break

        if proposed is None:
            sol_arts = [
                a for a in final_artifacts
                if a.type == ArtifactType.REVIEW
                and a.metadata.get("subtype") == "proposed_solution"
                and a.metadata.get("parsed_solution")
            ]
            if sol_arts:
                proposed = sol_arts[-1].metadata["parsed_solution"]

        cost = self.budget_monitor.current_cost
        real_faulty_id = self.faulty_experts[0].agent_id if self.faulty_experts else None

        self.bb_logger.log_episode_end(
            problem_id=puzzle_id,
            passed=passed,
            code=str(proposed) if proposed else "",
        )

        logger.info(
            f"Episode done: puzzle={puzzle_id} passed={passed} "
            f"fault={bool(self.faulty_experts)} auditor={bool(self.auditors)} "
            f"audit_correct={auditor_flagged_faulty} "
            f"cost=${cost:.4f}"
        )

        from ..utils.state_dump import auto_dump_episode_state
        auto_dump_episode_state(self.blackboard, puzzle_id, extra={
            "passed": passed, "proposed_solution": proposed,
            "ground_truth": ground_truth,
            "fault_injected": bool(self.faulty_experts),
            "auditor_used": bool(self.auditors),
            "auditor_flagged_faulty_agent": auditor_flagged_faulty,
            "auditor_suspect_id": auditor_suspect_id,
            "real_faulty_agent_id": real_faulty_id,
            "trust_scores": trust_scores,
            "n_contradictions_found": n_contradictions,
            "conflict_detected": conflict_detected,
            "conflict_resolved": conflict_resolved,
            "phase_timings": phase_timings,
        })

        return FaultInjectionEpisodeResult(
            puzzle_id=puzzle_id,
            passed=passed,
            proposed_solution=proposed,
            ground_truth=ground_truth,
            num_expert_deductions=num_deductions,
            num_synthesis_artifacts=len(solution_artifacts),
            fault_injected=bool(self.faulty_experts),
            auditor_used=bool(self.auditors),
            auditor_flagged_faulty_agent=auditor_flagged_faulty,
            auditor_suspect_id=auditor_suspect_id,
            real_faulty_agent_id=real_faulty_id,
            trust_scores=trust_scores,
            n_contradictions_found=n_contradictions,
            conflict_detected=conflict_detected,
            conflict_resolved=conflict_resolved,
            iterations=3 + (1 if conflict_detected else 0),
            cost_usd=cost,
            duration_s=duration_s,
            stop_reason="passed" if passed else "wrong_answer",
            phase_timings=phase_timings,
        )
