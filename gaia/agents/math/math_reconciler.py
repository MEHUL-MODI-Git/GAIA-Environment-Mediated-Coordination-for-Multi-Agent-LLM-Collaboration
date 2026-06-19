"""MathReconcilerAgent: resolves solver conflicts by auditing all reasoning chains.

Triggered only when the Aggregator detected a CONFLICT. The reconciler is the
most context-rich agent in the pipeline — it reads:
  - The original problem
  - ALL solver reasoning chains (not just their final answers)
  - The conflict summary (which solvers disagreed and by how much)

Unlike the MathSolverAgent (which is deliberately isolated), the Reconciler
needs full situational awareness. Its job is comparative: find the error in the
wrong solution(s) by contrasting them against the correct one(s). This is
impossible without seeing all chains side by side.

The Reconciler runs on the slower/more capable model (gpt-4.1) because auditing
multiple reasoning chains for subtle arithmetic errors is the hardest reasoning
task in the pipeline.

Output: ArtifactType.REVIEW with subtype="reconciled_solution"
  metadata["answer"] = final integer answer (authoritative)
  metadata["triggered_by_conflict"] = True
"""

from typing import List, Optional
from ...blackboard.models import Task, Artifact, ArtifactType, SignalType
from ...prompts.math.reconciler import MathReconcilerPrompts
from ..base import BaseAgent
from .math_solver import extract_final_answer  # reuse the same parser


class MathReconcilerAgent(BaseAgent):
    """Re-reasons from scratch after a solver conflict.

    Has access to:
    - The original problem (from task metadata)
    - All solver reasoning chains (from blackboard PLAN artifacts)
    - The conflict description (from the CONFLICT signal)

    This rich context lets the reconciler pinpoint errors rather than just
    blindly re-solving and hoping for a different answer.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "math_reconciler")
        super().__init__(**kwargs)
        self.prompts = MathReconcilerPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "math_reconcile"

    async def execute(self, task: Task) -> List[Artifact]:
        question = task.metadata.get("question", "")
        root_task_id = task.parent_id or task.task_id

        # ── Gather context from blackboard ────────────────────────────────────

        # 1. All solver reasoning chains
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        solver_artifacts = sorted(
            [
                a for a in all_artifacts
                if a.type == ArtifactType.PLAN
                and a.metadata.get("subtype") == "math_solution"
            ],
            key=lambda a: a.metadata.get("solver_index", 0),
        )

        # 2. Conflict summary from the CONFLICT signal
        conflict_signals = self.blackboard.get_signals(
            signal_type=SignalType.CONFLICT,
            resolved=False,
        )
        conflict_summary = "Solvers gave different answers"
        for sig in conflict_signals:
            if sig.task_id == root_task_id:
                conflict_summary = sig.metadata.get(
                    "conflict_summary", conflict_summary
                )
                break

        # ── Build prompt ──────────────────────────────────────────────────────

        solver_outputs = [
            (
                a.metadata.get("solver_label", f"Solver-{i+1}"),
                a.content,
            )
            for i, a in enumerate(solver_artifacts)
        ]

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {
                "role": "user",
                "content": self.prompts.format_user(
                    question=question,
                    conflict_summary=conflict_summary,
                    solver_outputs=solver_outputs,
                ),
            },
        ]

        # Reconciler always runs deterministically — we want the careful,
        # canonical answer, not creative variation
        response = await self.call_llm(messages, temperature=0.0)
        answer = extract_final_answer(response)

        self.logger.info(
            f"{self.name}: Reconciled answer={answer} "
            f"(conflict was: {conflict_summary})"
        )

        # Mark the CONFLICT signal as resolved
        for sig in conflict_signals:
            if sig.task_id == root_task_id:
                self.blackboard.resolve_signal(sig.signal_id)
                break

        artifact = Artifact(
            type=ArtifactType.REVIEW,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "reconciled_solution",
                "agent_name": self.name,
                "answer": answer,
                "triggered_by_conflict": True,
                "conflict_summary": conflict_summary,
                "num_solvers_reviewed": len(solver_artifacts),
            },
        )
        return [artifact]
