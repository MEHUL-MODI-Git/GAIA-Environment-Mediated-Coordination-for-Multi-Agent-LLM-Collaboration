"""MathVerifierAgent: checks the proposed answer against ground truth.

The verifier is the simplest agent in the pipeline. It finds the latest REVIEW
artifact (from the Reconciler if a conflict occurred, otherwise a direct REVIEW
posted by the Aggregator) or falls back to the majority PLAN answer, extracts
the integer, and compares it to the known ground truth.

No LLM call needed — integer comparison is deterministic and exact.

Output: Evidence posted to the blackboard with passed=True/False.
"""

from typing import List, Optional
from ...blackboard.models import (
    Task, Artifact, ArtifactType, Evidence,
)
from ..base import BaseAgent
from .math_solver import extract_final_answer


def _majority_answer(solver_artifacts) -> Optional[int]:
    """Return the answer that appears most often among solvers.

    If all answers differ (3-way conflict), returns None.
    """
    answers = [
        a.metadata.get("answer")
        for a in solver_artifacts
        if a.metadata.get("answer") is not None
    ]
    if not answers:
        return None
    from collections import Counter
    most_common_answer, count = Counter(answers).most_common(1)[0]
    return most_common_answer if count > 1 else answers[0]


class MathVerifierAgent(BaseAgent):
    """Verifies the final answer against ground truth.

    Answer resolution priority:
      1. Most recent REVIEW artifact (reconciled_solution — the authoritative answer)
      2. REVIEW artifact from aggregator (unanimous_answer)
      3. Majority vote from solver PLAN artifacts (fallback)

    This ordering ensures the Reconciler's answer always takes precedence when
    a conflict occurred.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "math_verifier")
        super().__init__(**kwargs)

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "math_verify"

    async def execute(self, task: Task) -> List[Artifact]:
        ground_truth: int = task.metadata.get("answer")
        root_task_id = task.parent_id or task.task_id

        # ── Find the best proposed answer ─────────────────────────────────────

        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        source_artifact = None

        # Priority 1: Reconciled REVIEW (conflict path — authoritative)
        reconciled = [
            a for a in all_artifacts
            if a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "reconciled_solution"
            and a.metadata.get("answer") is not None
        ]
        if reconciled:
            source_artifact = reconciled[-1]
            proposed_answer = source_artifact.metadata["answer"]
            source = "reconciled"
        else:
            # Priority 2: Unanimous REVIEW posted by the episode loop
            unanimous_reviews = [
                a for a in all_artifacts
                if a.type == ArtifactType.REVIEW
                and a.metadata.get("subtype") == "unanimous_answer"
                and a.metadata.get("answer") is not None
            ]
            if unanimous_reviews:
                source_artifact = unanimous_reviews[-1]
                proposed_answer = source_artifact.metadata["answer"]
                source = "unanimous"
            else:
                # Priority 3: Majority vote among raw solver answers (last resort)
                solver_artifacts = [
                    a for a in all_artifacts
                    if a.type == ArtifactType.PLAN
                    and a.metadata.get("subtype") == "math_solution"
                ]
                proposed_answer = _majority_answer(solver_artifacts)
                source = "majority_fallback"
                source_artifact = solver_artifacts[-1] if solver_artifacts else None

        passed = (proposed_answer is not None and proposed_answer == ground_truth)

        self.logger.info(
            f"{self.name}: proposed={proposed_answer} truth={ground_truth} "
            f"passed={passed} source={source}"
        )

        # Link Evidence to the artifact being verified so the episode loop
        # can collect it via get_evidence_for_artifact(artifact_id).
        evidence = Evidence(
            type="math_ground_truth",
            artifact_id=source_artifact.artifact_id if source_artifact else None,
            content=(
                f"Proposed: {proposed_answer} | Ground truth: {ground_truth} | "
                f"Result: {'PASS' if passed else 'FAIL'} | Source: {source}"
            ),
            passed=passed,
            metadata={
                "proposed_answer": proposed_answer,
                "ground_truth": ground_truth,
                "source": source,
                "passed": passed,
            },
        )
        self.blackboard.post_evidence(evidence)
        return []
