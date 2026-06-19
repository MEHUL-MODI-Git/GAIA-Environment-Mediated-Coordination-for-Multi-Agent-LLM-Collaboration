"""PuzzleCriticAgent: compares two synthesizer solutions, flags conflicts."""

import re
from typing import List, Optional
from ...blackboard.models import Task, Artifact, ArtifactType, Signal, SignalType, Claim
from ...prompts.puzzle.critic import PuzzleCriticPrompts
from ..base import BaseAgent


def extract_verdict(text: str) -> str:
    """Extract AGREE or CONFLICT from Critic's response."""
    if re.search(r"\bAGREE\b", text, re.IGNORECASE):
        return "AGREE"
    if re.search(r"\bCONFLICT\b", text, re.IGNORECASE):
        return "CONFLICT"
    return "UNKNOWN"


class PuzzleCriticAgent(BaseAgent):
    """Compares two proposed solutions from the Synthesizer agents.

    If both solutions agree → posts Claim confirming consensus.
    If they differ → posts CONFLICT Signal so episode loop can trigger re-synthesis.

    Output: Claim artifact + optional CONFLICT Signal.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "puzzle_critic")
        super().__init__(**kwargs)
        self.prompts = PuzzleCriticPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "critique"

    async def execute(self, task: Task) -> List[Artifact]:
        root_task_id = task.parent_id or task.task_id

        # Find both proposed solutions
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        solution_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
        ]

        if len(solution_artifacts) < 2:
            self.logger.warning(
                f"{self.name}: Only {len(solution_artifacts)} solution(s) found, "
                "need 2 to compare — skipping critique"
            )
            # No conflict to report; Verifier will use whatever solution exists
            return []

        sol1 = solution_artifacts[-2]
        sol2 = solution_artifacts[-1]

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(
                synth1_name=sol1.metadata.get("agent_name", "Synthesizer-1"),
                solution1=sol1.content,
                synth2_name=sol2.metadata.get("agent_name", "Synthesizer-2"),
                solution2=sol2.content,
            )},
        ]

        response = await self.call_llm(messages, temperature=0.0)
        verdict = extract_verdict(response)

        self.logger.info(f"{self.name}: Verdict = {verdict}")

        # Post claim to blackboard
        claim = Claim(
            task_id=root_task_id,
            author=self.agent_id,
            statement=f"Critic verdict: {verdict}\n\n{response}",
            confidence=1.0 if verdict == "AGREE" else 0.5,
        )
        self.blackboard.post_claim(claim)

        # If conflict, post a CONFLICT signal so the episode loop knows to re-synthesize
        if verdict == "CONFLICT":
            signal = Signal(
                type=SignalType.CONFLICT,
                task_id=root_task_id,
                description=f"Synthesizers disagree: {response[:200]}",
                severity=0.8,
                metadata={
                    "sol1_author": sol1.metadata.get("agent_name"),
                    "sol2_author": sol2.metadata.get("agent_name"),
                    "critic_response": response,
                },
            )
            self.blackboard.post_signal(signal)

        # Return no artifact (claims are posted directly)
        return []
