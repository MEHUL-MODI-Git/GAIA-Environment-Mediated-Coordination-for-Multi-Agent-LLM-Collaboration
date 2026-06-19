"""MathAggregatorAgent: checks solver consensus and raises CONFLICT if needed.

Runs in Phase 2 after all solvers have posted their answers. The aggregator
reads all solver artifacts, asks an LLM to extract the final integer from each
and compare them, then either:
  - Posts a REVIEW artifact ("UNANIMOUS: 42") and no signal, OR
  - Posts a CONFLICT Signal to the blackboard (triggering Phase 3 / Reconciler).

The aggregator knows it is reviewing multiple independent solvers. Unlike the
solvers, the aggregator needs this context to do its job: it must attribute
each answer to the correct solver so the Reconciler knows who to audit.

Output:
  Always: ArtifactType.PLAN with subtype="aggregator_verdict"
    metadata["unanimous"] = True/False
    metadata["agreed_answer"] = int (if unanimous) or None
    metadata["solver_answers"] = {solver_label: answer_int_or_none, ...}
    metadata["conflict_summary"] = str (if conflict)
"""

import re
from typing import List, Optional, Dict
from ...blackboard.models import (
    Task, Artifact, ArtifactType, Signal, SignalType,
)
from ...prompts.math.aggregator import MathAggregatorPrompts
from ..base import BaseAgent


def _parse_aggregator_output(text: str):
    """Parse the aggregator's UNANIMOUS or CONFLICT output.

    Returns (unanimous: bool, agreed_answer: int|None, conflict_summary: str).
    """
    text = text.strip()

    # Try UNANIMOUS: 42
    m = re.search(r"UNANIMOUS\s*:\s*([-]?\d[\d,]*)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return True, int(raw), ""
        except ValueError:
            pass

    # Try CONFLICT: Solver-1=42, Solver-2=35, ...
    m = re.search(r"CONFLICT\s*:(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        conflict_detail = m.group(1).strip().split("\n")[0]  # first line only
        return False, None, conflict_detail

    # Fallback: couldn't parse — treat as conflict to be safe
    return False, None, f"[Parse failed on aggregator output: {text[:200]}]"


class MathAggregatorAgent(BaseAgent):
    """Reads all solver answers and determines whether they agree.

    Knows about the multi-solver structure (needed to do its job), but does
    NOT re-solve the problem itself. Pure consensus detection.

    On conflict, posts a CONFLICT signal so the episode loop triggers the
    Reconciler. This is the mechanism that exercises Phase 3.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "math_aggregator")
        super().__init__(**kwargs)
        self.prompts = MathAggregatorPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "math_aggregate"

    async def execute(self, task: Task) -> List[Artifact]:
        question = task.metadata.get("question", "")
        root_task_id = task.parent_id or task.task_id

        # Read all solver artifacts from the blackboard
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        solver_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "math_solution"
        ]

        if not solver_artifacts:
            self.logger.warning(f"{self.name}: No solver artifacts found")
            return []

        # Sort by solver_index for consistent ordering
        solver_artifacts.sort(key=lambda a: a.metadata.get("solver_index", 0))

        # Build solver output list for the prompt
        solver_outputs = [
            (a.metadata.get("solver_label", f"Solver-{i+1}"), a.content)
            for i, a in enumerate(solver_artifacts)
        ]

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user",   "content": self.prompts.format_user(question, solver_outputs)},
        ]

        response = await self.call_llm(messages, temperature=0.0)
        unanimous, agreed_answer, conflict_summary = _parse_aggregator_output(response)

        # Collect solver answers for metadata
        solver_answers: Dict[str, Optional[int]] = {
            a.metadata.get("solver_label", f"Solver-{i+1}"): a.metadata.get("answer")
            for i, a in enumerate(solver_artifacts)
        }

        if unanimous:
            self.logger.info(
                f"{self.name}: UNANIMOUS answer={agreed_answer} "
                f"from {len(solver_artifacts)} solvers"
            )
        else:
            self.logger.info(
                f"{self.name}: CONFLICT detected — {conflict_summary} "
                f"(solver answers: {solver_answers})"
            )
            # Post CONFLICT signal so episode loop triggers Reconciler
            signal = Signal(
                type=SignalType.CONFLICT,
                task_id=root_task_id,
                description=f"Math solvers disagree: {conflict_summary}",
                severity=0.8,
                metadata={
                    "conflict_summary": conflict_summary,
                    "solver_answers": solver_answers,
                    "question": question,
                },
            )
            self.blackboard.post_signal(signal)

        artifact = Artifact(
            type=ArtifactType.PLAN,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "aggregator_verdict",
                "agent_name": self.name,
                "unanimous": unanimous,
                "agreed_answer": agreed_answer,
                "solver_answers": solver_answers,
                "conflict_summary": conflict_summary,
                "num_solvers": len(solver_artifacts),
            },
        )
        return [artifact]
