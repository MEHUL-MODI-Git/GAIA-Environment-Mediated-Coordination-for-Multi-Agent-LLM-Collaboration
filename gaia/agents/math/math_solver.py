"""MathSolverAgent: independently solves a math problem and posts its reasoning.

One of three solvers that run in parallel (Phase 1). Each solver receives the
same problem but operates at a different temperature, producing independent
reasoning chains. Importantly, solvers are NOT told other agents exist — this
prevents anchoring and ensures each reasoning chain is genuinely independent.

Output: ArtifactType.PLAN with subtype="math_solution".
  content = full reasoning chain (visible to Reconciler on conflict)
  metadata["answer"] = extracted integer answer (or None if parse failed)
"""

import re
from typing import List, Optional
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.math.solver import MathSolverPrompts
from ..base import BaseAgent


def extract_final_answer(text: str) -> Optional[int]:
    """Extract the integer from a line matching '**Final Answer: N**'.

    Looks for the last occurrence (in case the solver second-guesses itself).
    Returns the integer, or None if no parseable answer found.
    """
    # Match **Final Answer: 42** or **Final Answer: 42 dollars** etc.
    matches = re.findall(
        r"\*\*Final Answer\s*:\s*([-]?\d[\d,]*)\*\*",
        text,
        re.IGNORECASE,
    )
    if not matches:
        # Fallback: unformatted "Final Answer: 42" anywhere in text
        matches = re.findall(
            r"Final Answer\s*:\s*([-]?\d[\d,]*)",
            text,
            re.IGNORECASE,
        )
    if matches:
        # Take the last occurrence (most recent revision)
        raw = matches[-1].replace(",", "")
        try:
            return int(raw)
        except ValueError:
            return None
    return None


class MathSolverAgent(BaseAgent):
    """Solves a math problem independently and posts its reasoning + answer.

    Isolation design: the solver's system prompt does NOT mention other agents.
    Each solver should reach its answer through its own chain of thought,
    uninfluenced by what other solvers might produce.
    """

    def __init__(self, solver_index: int = 0, **kwargs):
        """
        Args:
            solver_index: 0, 1, or 2 — identifies which of the 3 parallel
                solvers this is. Used to tag the artifact so the Aggregator
                and Reconciler can identify which solver said what.
        """
        kwargs.setdefault("role", "math_solver")
        super().__init__(**kwargs)
        self.solver_index = solver_index
        self.prompts = MathSolverPrompts()

    def should_claim_task(self, task: Task) -> bool:
        expected = f"math_solve_{self.solver_index}"
        return task.metadata.get("task_type") == expected

    async def execute(self, task: Task) -> List[Artifact]:
        question = task.metadata.get("question", "")
        temperature = task.metadata.get("temperature", 0.0)
        root_task_id = task.parent_id or task.task_id

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(question)},
        ]

        response = await self.call_llm(messages, temperature=temperature)
        answer = extract_final_answer(response)

        solver_label = f"Solver-{self.solver_index + 1}"

        self.logger.info(
            f"{self.name} ({solver_label}): answer={answer} "
            f"({'parsed' if answer is not None else 'PARSE FAILED'})"
        )

        artifact = Artifact(
            type=ArtifactType.PLAN,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "math_solution",
                "solver_index": self.solver_index,
                "solver_label": solver_label,
                "agent_name": self.name,
                "answer": answer,          # int or None
                "temperature": temperature,
            },
        )
        return [artifact]
