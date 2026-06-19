"""ExpertAgent: reasons from a partial clue partition and posts deductions to blackboard."""

from typing import List
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.puzzle.expert import ExpertPrompts
from ..base import BaseAgent


class ExpertAgent(BaseAgent):
    """Analyzes one partition of clues and posts partial deductions.

    There are typically 4 Expert agents per puzzle (2 per partition).
    Agents on the same partition receive identical clues but may reason differently
    due to different temperature settings.

    Output: ArtifactType.PLAN artifact with subtype="partial_deduction".
    """

    def __init__(self, partition: str, **kwargs):
        """
        Args:
            partition: "A" or "B" — which clue partition this expert analyzes.
        """
        kwargs.setdefault("role", "expert")
        super().__init__(**kwargs)
        self.partition = partition
        self.prompts = ExpertPrompts()

    def should_claim_task(self, task: Task) -> bool:
        """Claim only the designated expert task for this partition.

        Also checks if this agent has already posted a deduction (prevents re-execution
        when the task is released and becomes open again).
        """
        expected_type = f"expert_{self.partition.lower()}"
        if task.metadata.get("task_type") != expected_type:
            return False
        # Check if this agent already posted a deduction for this puzzle
        root_task_id = task.parent_id or task.task_id
        existing = self.blackboard.get_artifacts_for_task(root_task_id)
        already_done = any(
            a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
            and a.author == self.agent_id
            for a in existing
        )
        return not already_done

    async def execute(self, task: Task) -> List[Artifact]:
        clues = task.metadata.get("clues", [])
        total_clues = task.metadata.get("total_clues", len(clues))

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(
                partition=self.partition,
                clues=clues,
                total_clues=total_clues,
            )},
        ]

        # temperature is passed via kwargs from the runner
        temperature = task.metadata.get("temperature", 0.2)
        response = await self.call_llm(messages, temperature=temperature)

        # Use the task's root_task_id so Synthesizer can find all deductions
        root_task_id = task.parent_id or task.task_id

        artifact = Artifact(
            type=ArtifactType.PLAN,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "partial_deduction",
                "partition": self.partition,
                "agent_name": self.name,
                "clues": clues,
            },
        )

        self.logger.info(
            f"{self.name} (Partition {self.partition}): "
            f"Posted partial deduction ({len(response)} chars)"
        )
        return [artifact]
