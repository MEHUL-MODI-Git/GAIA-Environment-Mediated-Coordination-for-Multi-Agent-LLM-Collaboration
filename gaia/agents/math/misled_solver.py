"""MisledSolverAgent: a math solver primed with a shared misleading heuristic.

Used in the Correlated Failure experiment (E3). Two MisledSolverAgents receive
the SAME misleading heuristic for a problem, so they produce the SAME wrong
answer — a deterministic correlated failure. A third clean MathSolverAgent
(no hint) produces the correct answer. This creates a 2-vs-1 majority that is
WRONG, which:
  - majority_vote condition: gets wrong (the failure mode we expose)
  - gaia condition: the Reconciler reads all 3 chains, spots the flawed shared
    heuristic, and sides with the clean dissenter (the mechanism we test)

Inherits MathSolverAgent so the Aggregator/Reconciler/Verifier and the
isinstance checks in the episode loop treat it uniformly. Output artifact is
identical (subtype="math_solution") with metadata["is_misled"]=True and the
hint recorded for analysis. The Reconciler is NEVER given the hint.
"""

from typing import List
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.math.misled_solver import MisledSolverPrompts
from .math_solver import MathSolverAgent, extract_final_answer


class MisledSolverAgent(MathSolverAgent):
    """Math solver that applies an injected (wrong) heuristic.

    Claims task_type == "math_solve_misled". The misleading hint is passed
    via task.metadata["misleading_hint"] by the episode loop.
    """

    def __init__(self, misled_index: int = 0, **kwargs):
        kwargs.setdefault("name", f"MisledSolver-{misled_index}")
        kwargs.setdefault("role", "misled_math_solver")
        # solver_index encodes misled slot as 100+index to avoid colliding
        # with clean solver indices 0/1/2.
        super().__init__(solver_index=100 + misled_index, **kwargs)
        self.misled_index = misled_index
        self.misled_prompts = MisledSolverPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == f"math_solve_misled_{self.misled_index}"

    async def execute(self, task: Task) -> List[Artifact]:
        question = task.metadata.get("question", "")
        hint = task.metadata.get("misleading_hint", "")
        temperature = task.metadata.get("temperature", 0.0)
        root_task_id = task.parent_id or task.task_id

        messages = [
            {"role": "system", "content": self.misled_prompts.SYSTEM},
            {"role": "user", "content": self.misled_prompts.format_user(question, hint)},
        ]

        response = await self.call_llm(messages, temperature=temperature)
        answer = extract_final_answer(response)

        self.logger.info(
            f"{self.name}: answer={answer} "
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
                "solver_label": f"Misled-Solver-{self.misled_index}",
                "agent_name": self.name,
                "answer": answer,
                "temperature": temperature,
                "is_misled": True,
                "injected_hint": hint,
            },
        )
        return [artifact]
