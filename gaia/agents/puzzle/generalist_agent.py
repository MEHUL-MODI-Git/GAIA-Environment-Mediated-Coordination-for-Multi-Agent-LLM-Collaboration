"""GeneralistAgent: homogeneous baseline for the Agent Scaling experiment (E8).

Unlike ExpertAgent (which sees a partition of clues), the Generalist sees ALL
clues and attempts a complete solution independently. Multiple Generalists run
in parallel and a majority vote (computed outside the blackboard) selects the
final answer.

This tests whether GAIA's accuracy gains come from:
  (a) Role specialization + coordination (ExpertA + ExpertB + Synthesizer), or
  (b) Simply having more agents.

Hypothesis (b) is false — adding N identical agents adds no new information
because they all process the same clues with the same prompt. The benefit
comes from role diversity and blackboard coordination.

Output: ArtifactType.REVIEW with subtype="generalist_solution". The runner
collects all generalist artifacts for a puzzle and majority-votes the answers.
"""

from typing import List
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.puzzle.generalist import GeneralistPrompts
from ..base import BaseAgent


class GeneralistAgent(BaseAgent):
    """A single solver that reads ALL clues and outputs a complete solution.

    Multiple instances are deployed in parallel (homogeneous pool). Each one
    independently produces a solution; majority vote selects the final answer.
    There is NO coordination between Generalists — this is the "naive horizontal
    scaling" baseline.

    Args:
        generalist_index: 0-indexed position in the pool (so each instance
            claims a distinct task slot and the runner can distinguish outputs).
        people: list of person names in the puzzle (passed so the prompt can
            adapt to 4×3 or 6×4 puzzles).
        attributes: dict {attr_name: [possible_values]} describing the puzzle.
    """

    def __init__(
        self,
        generalist_index: int,
        people: list,
        attributes: dict,
        **kwargs,
    ):
        kwargs.setdefault("name", f"Generalist-{generalist_index}")
        kwargs.setdefault("role", "generalist")
        super().__init__(**kwargs)
        self.generalist_index = generalist_index
        self.people = people
        self.attributes = attributes
        self.prompts = GeneralistPrompts()

    def should_claim_task(self, task: Task) -> bool:
        expected = f"generalist_solve_{self.generalist_index}"
        if task.metadata.get("task_type") != expected:
            return False
        # Prevent re-execution if this agent already posted its solution
        root_task_id = task.parent_id or task.task_id
        existing = self.blackboard.get_artifacts_for_task(root_task_id)
        already_done = any(
            a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "generalist_solution"
            and a.author == self.agent_id
            for a in existing
        )
        return not already_done

    async def execute(self, task: Task) -> List[Artifact]:
        clues = task.metadata.get("clues", [])
        temperature = task.metadata.get("temperature", 0.2)
        root_task_id = task.parent_id or task.task_id

        messages = [
            {"role": "system", "content": self.prompts.format_system(len(self.people))},
            {"role": "user", "content": self.prompts.format_user(
                clues=clues,
                people=self.people,
                attributes=self.attributes,
            )},
        ]

        response = await self.call_llm(messages, temperature=temperature)

        artifact = Artifact(
            type=ArtifactType.REVIEW,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "generalist_solution",
                "agent_name": self.name,
                "generalist_index": self.generalist_index,
                "temperature": temperature,
                "n_clues_seen": len(clues),
            },
        )

        self.logger.info(
            f"{self.name}: Posted generalist solution "
            f"(saw all {len(clues)} clues, {len(response)} chars)"
        )
        return [artifact]
