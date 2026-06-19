"""Episode loop - the 7-step GAIA coordination cycle"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    Task,
    TaskStatus,
    Artifact,
    ArtifactType,
    Policy,
    SignalType,
)
from ..micro_checkers import run_micro_checkers
from ..agents.base import BaseAgent
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger
from ..utils.budget_monitor import BudgetMonitor
from ..utils.blackboard_logger import BlackboardLogger
from .scheduler import Scheduler
from pydantic import BaseModel

logger = get_logger("episode")


class EpisodeResult(BaseModel):
    """Result from running one episode"""

    task_id: str
    passed: bool
    code: str = ""
    iterations: int = 0
    artifacts_created: int = 0
    conflicts_detected: int = 0
    branches_created: int = 0
    metadata: Dict[str, Any] = {}


class EpisodeLoop:
    """7-step GAIA episode loop for one HumanEval problem"""

    def __init__(
        self,
        blackboard: Blackboard,
        agents: List[BaseAgent],
        metrics: MetricsCollector,
        policy: Policy,
        budget_monitor: Optional[BudgetMonitor] = None,
    ):
        self.blackboard = blackboard
        self.agents = agents
        self.metrics = metrics
        self.policy = policy
        self.scheduler = Scheduler(blackboard)

        # Budget monitoring (default: $0.30 per problem)
        self.budget_monitor = budget_monitor or BudgetMonitor(
            max_cost_per_problem=0.30,
            max_iterations=policy.max_iterations,
            max_llm_calls=50,
        )

        # Access to blackboard logger for episode-level events
        self.bb_logger = blackboard.logger

    async def run_episode(self, problem: Dict[str, Any]) -> EpisodeResult:
        """Run complete episode for one HumanEval problem

        Implements the 7-step GAIA loop:
        1. Initialize - post root task
        2. Decompose - planner creates subtasks (optional for HumanEval)
        3. Self-assignment - agents poll and claim
        4. Production - agents execute and create artifacts
        5. Detect signals - find conflicts, uncertainty
        6. Resolve - conflict-as-task, branch-and-merge
        7. Verify - check acceptance criteria

        Args:
            problem: HumanEval problem dict

        Returns:
            EpisodeResult with outcome and metrics
        """
        task_id = problem["task_id"]
        start_time = datetime.utcnow()

        # Log episode start
        self.bb_logger.log_episode_start(problem_id=task_id)
        logger.info(f"=== Starting episode for {task_id} ===")

        # Reset budget monitor for this problem
        self.budget_monitor.reset()

        # ===== Step 0: Run micro-checkers (zero LLM cost) =====
        micro_report = run_micro_checkers(problem["prompt"])
        micro_warnings = micro_report.to_prompt_block() if micro_report.any_triggered else ""
        if micro_report.any_triggered:
            logger.info(f"Micro-checkers triggered: {micro_report.triggered_names}")

        # ===== Step 1: Initialize =====
        root_task = Task(
            title=f"Solve {problem['entry_point']}",
            description=problem["prompt"],
            acceptance_criteria="All unit tests must pass",
            metadata={
                "test": problem["test"],
                "entry_point": problem["entry_point"],
                "task_id_humaneval": task_id,
                "task_type": "code_implementation",
                "micro_checker_warnings": micro_warnings,
                "micro_risk_flags": micro_report.risk_flags(),
            },
        )
        self.blackboard.post_task(root_task)
        logger.info(f"Step 1: Posted root task {root_task.task_id}")

        # ===== Failure pattern tracker =====
        # Maps error fingerprint -> count. Used to detect looping on same error.
        failure_pattern: Dict[str, int] = {}
        branch_triggered = False  # Only trigger branch-and-merge once per episode
        prev_code_hash: Optional[int] = None  # For stall detection (local, reset per episode)

        # ===== Steps 2-6: Iterative loop =====
        for iteration in range(self.policy.max_iterations):
            # Log iteration start
            self.bb_logger.log_iteration(iteration_num=iteration + 1, start=True)
            logger.info(f"\n--- Iteration {iteration + 1} ---")
            self.metrics.record_iteration()

            # Check budget BEFORE starting iteration
            can_continue = self.budget_monitor.record_iteration()
            if not can_continue:
                logger.error("❌ Budget iteration limit exceeded, stopping episode")
                break

            should_continue, stop_reason = self.budget_monitor.should_continue()
            if not should_continue:
                logger.error(f"❌ Budget limit exceeded: {stop_reason}")
                latest_code = self.blackboard.get_latest_artifact(
                    root_task.task_id, ArtifactType.CODE
                )
                self.bb_logger.log_episode_end(
                    problem_id=task_id,
                    passed=False,
                    code=latest_code.content if latest_code else ""
                )
                return EpisodeResult(
                    task_id=task_id,
                    passed=False,
                    code=latest_code.content if latest_code else "",
                    iterations=iteration,
                    metadata={
                        "stop_reason": "budget_exceeded",
                        "budget_summary": self.budget_monitor.get_summary()
                    }
                )

            # Step 2: Planner (once, on iteration 1, for complex problems)
            # Planner posts a PLAN artifact with strategic hints; Coder reads it on first attempt.
            if iteration == 0:
                await self._run_planner(root_task)

            # Step 3 & 4: Self-assignment and Production
            # Run all agents in parallel
            await self._run_agents_parallel()

            # Log iteration end
            self.bb_logger.log_iteration(iteration_num=iteration + 1, start=False)

            # Step 4: Check for passing solution
            latest_code = self.blackboard.get_latest_artifact(
                root_task.task_id, ArtifactType.CODE
            )

            # Stall detection: if code didn't change, trigger branch-and-merge immediately
            new_code_hash = hash(latest_code.content.strip()) if latest_code else None
            stalled = (new_code_hash is not None and new_code_hash == prev_code_hash)
            prev_code_hash = new_code_hash
            if stalled and self.policy.branch_trigger_on_failure and not branch_triggered:
                logger.warning("Step 4: Code unchanged — stalled refinement, triggering branch-and-merge")
                branch_triggered = True
                success = await self._branch_and_merge(root_task, problem)
                if success:
                    winning_code = self.blackboard.get_latest_artifact(
                        root_task.task_id, ArtifactType.CODE
                    )
                    self.bb_logger.log_episode_end(
                        problem_id=task_id,
                        passed=True,
                        code=winning_code.content if winning_code else ""
                    )
                    return EpisodeResult(
                        task_id=task_id,
                        passed=True,
                        code=winning_code.content if winning_code else "",
                        iterations=iteration + 1,
                        branches_created=self.policy.branch_max_parallel,
                    )

            if latest_code:
                # Check if we have test evidence
                evidence_list = self.blackboard.get_evidence_for_artifact(
                    latest_code.artifact_id
                )

                if evidence_list and evidence_list[0].passed:
                    # Success!
                    logger.info(f"✓ Episode complete: Tests passed!")
                    self.blackboard.complete_task(root_task.task_id, [latest_code])

                    # Log episode end
                    self.bb_logger.log_episode_end(
                        problem_id=task_id,
                        passed=True,
                        code=latest_code.content
                    )

                    return EpisodeResult(
                        task_id=task_id,
                        passed=True,
                        code=latest_code.content,
                        iterations=iteration + 1,
                        artifacts_created=len(
                            self.blackboard.get_artifacts_for_task(root_task.task_id)
                        ),
                        metadata={
                            "budget_summary": self.budget_monitor.get_summary(),
                            "episode_summary": self.bb_logger.get_episode_summary()
                        }
                    )

            # Step 5: Detect conflicts & uncertainty
            signals = self.blackboard.detect_signals()
            conflicts = [s for s in signals if s.type == SignalType.CONFLICT]

            if conflicts:
                self.metrics.record_conflict()
                logger.info(f"Step 5: Detected {len(conflicts)} conflicts")

                # Track failure pattern (fingerprint based on latest test output)
                failure_count = root_task.metadata.get("failure_count", 0)
                for conflict in conflicts:
                    self.bb_logger.log_conflict_detected(
                        task_id=root_task.task_id,
                        conflict_type=conflict.metadata.get("conflict_type", "unknown"),
                        failure_count=failure_count
                    )
                    # Fingerprint on stable test stderr, not varying Critic text
                    evidence_id = conflict.metadata.get("evidence_id")
                    if evidence_id:
                        ev = self.blackboard.storage.get_evidence(evidence_id)
                        fingerprint = ev.content[:120].strip() if ev else conflict.description[:80]
                    else:
                        fingerprint = conflict.description[:80].strip()
                    failure_pattern[fingerprint] = failure_pattern.get(fingerprint, 0) + 1
                    repeat_count = failure_pattern[fingerprint]
                    if repeat_count >= 3:
                        logger.warning(
                            f"Step 5: Same error pattern repeated {repeat_count}x — "
                            "loop detected, will trigger branch-and-merge"
                        )

            # Step 6: Resolve conflicts
            if conflicts:
                # Feature E: Conflict-as-task
                await self._resolve_conflicts(root_task, conflicts)

                # Feature F: Branch-and-merge — progressive escalation
                # Trigger only once per episode, after enough failures to justify cost:
                # - policy flag must be enabled, AND
                # - not already triggered this episode, AND
                # - either: a repeat loop detected OR failure_count > escalation threshold
                failure_count = root_task.metadata.get("failure_count", 0)
                loop_detected = any(v >= 2 for v in failure_pattern.values())
                escalation_threshold = 3  # Escalate after 3 failures
                should_branch = (
                    self.policy.branch_trigger_on_failure
                    and not branch_triggered
                    and (loop_detected or failure_count >= escalation_threshold)
                )
                if should_branch:
                    branch_triggered = True
                    logger.info(
                        f"Step 6: Triggering branch-and-merge "
                        f"(failure_count={failure_count}, loop={loop_detected})"
                    )
                    success = await self._branch_and_merge(root_task, problem)
                    if success:
                        winning_code = self.blackboard.get_latest_artifact(
                            root_task.task_id, ArtifactType.CODE
                        )
                        self.bb_logger.log_episode_end(
                            problem_id=task_id,
                            passed=True,
                            code=winning_code.content if winning_code else ""
                        )
                        return EpisodeResult(
                            task_id=task_id,
                            passed=True,
                            code=winning_code.content if winning_code else "",
                            iterations=iteration + 1,
                            branches_created=self.policy.branch_max_parallel,
                        )

        # ===== Step 7: Final verification =====
        logger.info(f"✗ Episode ended: Max iterations reached")

        latest_code = self.blackboard.get_latest_artifact(root_task.task_id, ArtifactType.CODE)
        code = latest_code.content if latest_code else ""

        # Log episode end
        self.bb_logger.log_episode_end(
            problem_id=task_id,
            passed=False,
            code=code
        )

        return EpisodeResult(
            task_id=task_id,
            passed=False,
            code=code,
            iterations=self.policy.max_iterations,
            artifacts_created=len(self.blackboard.get_artifacts_for_task(root_task.task_id)),
            metadata={
                "stop_reason": "max_iterations",
                "budget_summary": self.budget_monitor.get_summary(),
                "episode_summary": self.bb_logger.get_episode_summary()
            }
        )

    async def _run_planner(self, root_task: Task):
        """Step 2: Run the Planner agent (if present) for complex problems.

        The Planner posts a PLAN artifact with strategic hints. It decides
        internally whether the problem is complex enough to warrant analysis.
        Runs silently — no error if no planner agent is present.
        """
        from ..agents.planner import PlannerAgent
        planner = next((a for a in self.agents if isinstance(a, PlannerAgent)), None)
        if planner is None:
            return
        try:
            await planner.run_loop(max_iterations=1)
            plan = self.blackboard.get_latest_artifact(root_task.task_id, ArtifactType.PLAN)
            if plan:
                logger.info(f"Step 2: Planner posted PLAN ({len(plan.content)} chars)")
        except Exception as exc:
            logger.warning(f"Step 2: Planner error (non-fatal): {exc}")

    async def _run_agents_parallel(self):
        """Run all non-Planner agents in parallel (each does one task)."""
        from ..agents.planner import PlannerAgent
        # Planner runs separately in step 2 (once only); exclude from main parallel loop
        active_agents = [a for a in self.agents if not isinstance(a, PlannerAgent)]
        tasks = [agent.run_loop(max_iterations=1) for agent in active_agents]
        await asyncio.gather(*tasks)

    async def _resolve_conflicts(self, root_task: Task, conflicts: list):
        """Feature E: Create conflict resolution tasks with intelligent feedback"""
        for conflict in conflicts:
            # Create a fix task with feedback
            latest_code = self.blackboard.get_latest_artifact(
                root_task.task_id, ArtifactType.CODE
            )

            # Track failure count for this task
            failure_count = root_task.metadata.get("failure_count", 0) + 1
            root_task.metadata["failure_count"] = failure_count
            self.blackboard.storage.update_task(root_task)

            # Get detailed feedback from Critic after verification failure
            feedback = conflict.description
            if "test_result" in conflict.metadata and conflict.metadata.get("conflict_type") == "test_failure":
                # This is a verification failure - get Critic feedback
                critic = next((a for a in self.agents if a.role == "critic"), None)
                if critic and latest_code:
                    # Get test output from evidence
                    evidence_id = conflict.metadata.get("evidence_id")
                    if evidence_id:
                        evidence = self.blackboard.storage.get_evidence(evidence_id)
                        if evidence:
                            # Get detailed feedback from Critic
                            critic_feedback = await critic.provide_feedback_after_verification(
                                root_task, latest_code.content, evidence.content
                            )
                            feedback = f"{conflict.description}\n\nCritic Feedback:\n{critic_feedback}"

            # Check if we need EdgeCase agent (3+ repeated failures)
            if failure_count >= 3:
                logger.warning(
                    f"Task {root_task.task_id} has failed {failure_count} times - "
                    "triggering EdgeCase agent"
                )

                # Create EdgeCase analysis task
                edge_case_task = Task(
                    parent_id=root_task.task_id,
                    title="Diagnose repeated failures",
                    description=root_task.description,
                    metadata={
                        **root_task.metadata,
                        "previous_code": latest_code.content if latest_code else "",
                        "failure_count": failure_count,
                        "error_pattern": conflict.description,
                        "task_type": "edge_case_analysis",
                    },
                    priority=root_task.priority + 2.0,  # Very high priority
                )
                self.blackboard.post_task(edge_case_task)
                logger.info(f"Created edge case analysis task: {edge_case_task.task_id}")

            # Create fix task
            fix_task = Task(
                parent_id=root_task.task_id,
                title=f"Fix attempt #{failure_count}",
                description=root_task.description,  # Same problem
                metadata={
                    **root_task.metadata,
                    "previous_code": latest_code.content if latest_code else "",
                    "feedback": feedback,
                    "task_type": "code_fix",
                    "iteration": failure_count,
                },
                priority=root_task.priority + 1.0,  # Higher priority
            )

            self.blackboard.post_task(fix_task)
            logger.info(f"Created fix task #{failure_count}: {fix_task.task_id}")

    async def _branch_and_merge(self, root_task: Task, problem: Dict) -> bool:
        """Feature F: Fork blackboard, try parallel solutions, merge winner.

        Automatically extracts the latest failure summary from Evidence and passes
        the Planner agent so branches receive problem-specific algorithmic approaches.
        """
        logger.info("Branching: Creating parallel solution attempts")

        # Extract latest test failure output to give Planner context
        failure_summary = ""
        all_conflict_signals = self.blackboard.get_signals(signal_type=SignalType.CONFLICT)
        conflicts = [s for s in all_conflict_signals if s.task_id == root_task.task_id]
        for conflict in conflicts:
            ev_id = conflict.metadata.get("evidence_id")
            if ev_id:
                ev = self.blackboard.storage.get_evidence(ev_id)
                if ev:
                    failure_summary = ev.content[:500]
                    break
        if not failure_summary:
            # Fall back to latest code evidence
            latest_code = self.blackboard.get_latest_artifact(root_task.task_id, ArtifactType.CODE)
            if latest_code:
                evs = self.blackboard.get_evidence_for_artifact(latest_code.artifact_id)
                if evs:
                    failure_summary = evs[0].content[:500]

        # Find Planner agent for problem-specific approach generation
        from ..agents.planner import PlannerAgent
        planner_agent = next((a for a in self.agents if isinstance(a, PlannerAgent)), None)

        # Import here to avoid circular dependency
        from ..resolution.branch_merge import BranchManager

        branch_manager = BranchManager(
            self.blackboard, self.agents, self.metrics, self.budget_monitor
        )
        winner_code = await branch_manager.branch_and_merge(
            root_task, problem,
            n_branches=self.policy.branch_max_parallel,
            planner=planner_agent,
            failure_summary=failure_summary,
        )

        return winner_code is not None
