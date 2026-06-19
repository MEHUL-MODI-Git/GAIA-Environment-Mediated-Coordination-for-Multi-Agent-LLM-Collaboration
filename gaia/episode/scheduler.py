"""Scheduler for agent coordination (Feature C: Self-Assignment)"""

from typing import List
from ..blackboard.blackboard import Blackboard
from ..blackboard.models import Task, TaskStatus
from ..agents.base import BaseAgent
from ..utils.logging import get_logger

logger = get_logger("scheduler")


class Scheduler:
    """Manage agent self-assignment and task scheduling"""

    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard

    def get_available_tasks(self, agent: BaseAgent) -> List[Task]:
        """Get tasks available for an agent

        Args:
            agent: Agent requesting tasks

        Returns:
            List of tasks the agent can claim
        """
        all_available = self.blackboard.get_open_tasks(available_only=True)

        # Filter by agent tier if routing rules exist
        policy = self.blackboard.policy
        if policy.routing_rules:
            filtered = []
            for task in all_available:
                task_type = task.metadata.get("task_type", "default")
                required_tier = policy.routing_rules.get(task_type, agent.tier.value)

                if required_tier == agent.tier.value:
                    filtered.append(task)

            return filtered

        return all_available

    def should_spawn(self, current_agents: int) -> bool:
        """Check if we should spawn more agents (Feature D)

        Args:
            current_agents: Current number of active agents

        Returns:
            True if spawning recommended
        """
        policy = self.blackboard.policy

        # Check backlog
        open_tasks = self.blackboard.get_open_tasks(available_only=True)
        backlog_size = len(open_tasks)

        if backlog_size > policy.spawn_threshold and current_agents < policy.max_agents:
            logger.info(
                f"Spawn recommended: backlog={backlog_size}, "
                f"threshold={policy.spawn_threshold}"
            )
            return True

        return False

    def detect_stale_tasks(self) -> List[Task]:
        """Find tasks with expired leases"""
        stale = []
        open_tasks = self.blackboard.storage.query_tasks(status=TaskStatus.OPEN)

        for task in open_tasks:
            if task.lease and task.lease.is_expired():
                stale.append(task)
                # Auto-release stale tasks
                self.blackboard.release_task(task.task_id)
                logger.warning(f"Released stale task: {task.task_id}")

        return stale
