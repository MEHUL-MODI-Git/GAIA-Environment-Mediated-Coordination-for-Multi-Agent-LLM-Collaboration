"""SynthesizerAgent: merges all expert deductions into a complete solution."""

import re
from typing import List, Dict, Optional
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.puzzle.synthesizer import SynthesizerPrompts
from ..base import BaseAgent

PEOPLE = ["Alice", "Bob", "Carol", "Dave"]
ATTRIBUTES = ["job", "pet", "drink"]


def parse_solution_from_text(text: str) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Extract the structured solution from the agent's free-text response.

    Looks for lines like:
        Alice:  job=doctor, pet=cat, drink=coffee
    Returns dict[person][attr] = value, or None if parsing fails.
    """
    solution = {}
    for person in PEOPLE:
        # Match: Alice: job=doctor, pet=cat, drink=coffee (flexible whitespace/punctuation)
        pattern = (
            rf"{person}\s*:\s*"
            rf"job\s*=\s*(\w+)\s*[,;]?\s*"
            rf"pet\s*=\s*(\w+)\s*[,;]?\s*"
            rf"drink\s*=\s*(\w+)"
        )
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            solution[person] = {
                "job":   m.group(1).strip().lower(),
                "pet":   m.group(2).strip().lower(),
                "drink": m.group(3).strip().lower(),
            }

    if len(solution) == len(PEOPLE):
        return solution
    return None


class SynthesizerAgent(BaseAgent):
    """Reads all expert deductions from the blackboard and produces a full solution.

    Two Synthesizers run independently (different temperatures).
    Both write REVIEW artifacts; the Critic then compares them.

    Output: ArtifactType.REVIEW artifact with subtype="proposed_solution".
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "synthesizer")
        super().__init__(**kwargs)
        self.prompts = SynthesizerPrompts()

    def should_claim_task(self, task: Task) -> bool:
        if task.metadata.get("task_type") != "synthesize":
            return False
        # Don't re-execute if this agent already produced a solution for the same synthesize task
        root_task_id = task.parent_id or task.task_id
        existing = self.blackboard.get_artifacts_for_task(root_task_id)
        task_title = task.title
        already_done = any(
            a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
            and a.author == self.agent_id
            and a.metadata.get("source_task", "") == task.task_id
            for a in existing
        )
        return not already_done

    async def execute(self, task: Task) -> List[Artifact]:
        root_task_id = task.parent_id or task.task_id

        # Read all partial deductions from blackboard
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        deduction_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
        ]

        if not deduction_artifacts:
            self.logger.warning(f"{self.name}: No expert deductions found — cannot synthesize")
            return []

        # Build list of (agent_name, partition, content)
        expert_deductions = [
            (
                a.metadata.get("agent_name", a.author),
                a.metadata.get("partition", "?"),
                a.content,
            )
            for a in deduction_artifacts
        ]

        self.logger.info(
            f"{self.name}: Synthesizing from {len(expert_deductions)} expert deductions "
            f"(partitions: {sorted(set(p for _, p, _ in expert_deductions))})"
        )

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(expert_deductions)},
        ]

        temperature = task.metadata.get("temperature", 0.1)
        response = await self.call_llm(messages, temperature=temperature)

        # Extract structured solution for downstream use
        parsed = parse_solution_from_text(response)

        artifact = Artifact(
            type=ArtifactType.REVIEW,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "proposed_solution",
                "agent_name": self.name,
                "parsed_solution": parsed,  # dict or None
                "num_experts_read": len(expert_deductions),
                "source_task": task.task_id,
            },
        )

        self.logger.info(
            f"{self.name}: Posted proposed solution "
            f"(parsed={'ok' if parsed else 'FAILED to parse'})"
        )
        return [artifact]
