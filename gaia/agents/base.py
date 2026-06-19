"""Base agent with blackboard integration (Feature C: Self-Assignment)"""

from abc import ABC, abstractmethod
from typing import List, Optional
import uuid
import time

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import Task, Artifact
from ..llms.base import BaseLLM, ModelTier
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger
from ..utils.budget_monitor import BudgetMonitor

logger = get_logger("agents")


class BaseAgent(ABC):
    """Base agent with self-assignment from blackboard

    Key difference from AgentVerse: GAIA agents poll and claim tasks from the
    blackboard (Feature C), rather than being called by a central orchestrator.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "BaseAgent",
        role: str = "generic",
        tier: ModelTier = ModelTier.FAST,
        llm: Optional[BaseLLM] = None,
        blackboard: Optional[Blackboard] = None,
        metrics: Optional[MetricsCollector] = None,
        budget_monitor: Optional[BudgetMonitor] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.role = role
        self.tier = tier
        self.llm = llm
        self.blackboard = blackboard
        self.metrics = metrics
        self.budget_monitor = budget_monitor
        self.logger = get_logger(f"agent.{self.role}")

    async def run_loop(self, max_iterations: int = 10) -> int:
        """Main agent loop: poll -> claim -> execute -> post results

        Returns:
            Number of tasks completed
        """
        tasks_completed = 0

        for iteration in range(max_iterations):
            # Step 1: Poll for available task
            task = self.blackboard.poll_task(self.agent_id, self.tier.value)

            if task is None:
                # No more work available
                self.logger.info(f"{self.name}: No tasks available, stopping")
                break

            # Step 1.5: Check if this agent should claim this task
            if not self.should_claim_task(task):
                # Agent-specific logic says skip this task
                continue

            # Step 2: Claim the task
            success = self.blackboard.claim_task(self.agent_id, task.task_id)
            if not success:
                # Another agent claimed it
                self.logger.info(f"{self.name}: Task {task.task_id} claimed by another agent")
                continue

            self.logger.info(f"{self.name}: Claimed task {task.task_id} - {task.title}")

            # Step 3: Execute the task
            try:
                # Log execution start
                start_time = time.time()
                self.blackboard.logger.log_agent_execute(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    start=True
                )

                artifacts = await self.execute(task)

                # Log execution end
                duration_ms = (time.time() - start_time) * 1000
                self.blackboard.logger.log_agent_execute(
                    agent_id=self.agent_id,
                    task_id=task.task_id,
                    start=False,
                    duration_ms=duration_ms
                )

                # Step 4: Post artifacts and release task
                # Post artifacts first
                for artifact in artifacts:
                    self.blackboard.post_artifact(artifact)

                # Release task so other agents (Verifier) can claim it
                # Only Verifier will complete tasks based on test results
                self.blackboard.release_task(task.task_id)

                tasks_completed += 1

                self.logger.info(
                    f"{self.name}: Posted {len(artifacts)} artifact(s) for task {task.task_id}, released task"
                )

            except Exception as e:
                # Task failed
                self.logger.error(f"{self.name}: Task {task.task_id} failed: {str(e)}")
                self.blackboard.fail_task(task.task_id, str(e))

        return tasks_completed

    def should_claim_task(self, task: Task) -> bool:
        """Check if this agent should claim a task

        Override in subclasses to add agent-specific logic.
        Default: claim all available tasks.

        Args:
            task: Task to evaluate

        Returns:
            True if agent should claim task, False to skip
        """
        return True

    @abstractmethod
    async def execute(self, task: Task) -> List[Artifact]:
        """Execute a task and produce artifacts

        Args:
            task: Task to execute

        Returns:
            List of artifacts produced
        """

    async def call_llm(self, messages: List[dict], **kwargs) -> str:
        """Helper to call LLM and track metrics"""
        if not self.llm:
            raise ValueError(f"{self.name}: No LLM configured")

        result = await self.llm.agenerate(messages, **kwargs)

        # Log LLM call with full metrics
        if self.blackboard:
            self.blackboard.logger.log_llm_call(
                agent_id=self.agent_id,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms
            )

        # Track metrics in MetricsCollector
        if self.metrics:
            self.metrics.record_llm_call(
                agent_id=self.agent_id,
                provider=result.provider,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )

        # Check budget if monitor is available
        if self.budget_monitor:
            can_continue = self.budget_monitor.record_llm_call(result.cost_usd)
            if not can_continue:
                from ..utils.budget_monitor import BudgetExceededException
                raise BudgetExceededException(
                    f"Budget exceeded after LLM call: ${result.cost_usd:.4f}"
                )

        return result.content
