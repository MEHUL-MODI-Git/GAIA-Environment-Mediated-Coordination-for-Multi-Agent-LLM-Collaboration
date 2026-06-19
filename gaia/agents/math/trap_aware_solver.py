"""TrapAwareSolverAgent: a math solver that audits its own answer against
common systematic error patterns before committing.

Used in the Correlated Failure experiment (E3). The agent pool has:
  - 2 × MathSolverAgent  (standard, prone to traps)
  - 1 × TrapAwareSolverAgent  (self-audits for boundary/rate/sequence traps)

On trap problems, the 2 standard solvers both fall into the same error,
creating a 2-vs-1 majority disagreement. The TrapAwareSolverAgent provides
the dissenting correct answer, which the Reconciler should side with after
auditing all three reasoning chains.

Output format: identical to MathSolverAgent — ArtifactType.PLAN with
subtype="math_solution". The Aggregator and Reconciler work with both types
transparently. The only difference is solver_label="TrapAware-Solver" and
an additional metadata["trap_audit"] field with the self-check note.
"""

from typing import List, Optional
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.math.trap_aware_solver import TrapAwareSolverPrompts
from .math_solver import MathSolverAgent, extract_final_answer
import re


class TrapAwareSolverAgent(MathSolverAgent):
    """Math solver with built-in self-audit for systematic error traps.

    Inherits MathSolverAgent so isinstance checks in the episode loop
    include it in the solver pool automatically. Overrides:
      - should_claim_task: claims "math_solve_trap_aware" tasks
      - execute: uses TrapAwareSolverPrompts and extracts trap audit note
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "TrapAwareSolver")
        kwargs.setdefault("role", "trap_aware_math_solver")
        # solver_index=-1 so it doesn't collide with 0/1/2 slots
        super().__init__(solver_index=-1, **kwargs)
        self.trap_prompts = TrapAwareSolverPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "math_solve_trap_aware"

    def _extract_trap_audit(self, text: str) -> str:
        """Pull out the TRAP AUDIT section from the solver's response."""
        m = re.search(r"TRAP AUDIT(.*?)(\*\*Final Answer|\Z)", text,
                      re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()[:500]
        return ""

    async def execute(self, task: Task) -> List[Artifact]:
        question = task.metadata.get("question", "")
        temperature = task.metadata.get("temperature", 0.0)
        root_task_id = task.parent_id or task.task_id

        messages = [
            {"role": "system", "content": self.trap_prompts.SYSTEM},
            {"role": "user", "content": self.trap_prompts.format_user(question)},
        ]

        response = await self.call_llm(messages, temperature=temperature)
        answer = extract_final_answer(response)
        trap_audit = self._extract_trap_audit(response)

        self.logger.info(
            f"{self.name}: answer={answer} trap_audit_len={len(trap_audit)} "
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
                "solver_label": "TrapAware-Solver",
                "agent_name": self.name,
                "answer": answer,
                "temperature": temperature,
                "trap_audit": trap_audit,
                "is_trap_aware": True,
            },
        )
        return [artifact]
