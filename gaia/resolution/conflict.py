"""Conflict detection and resolution (Feature E: Conflict-as-Task)"""

from typing import List, Optional
from datetime import datetime

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import (
    Task,
    Signal,
    SignalType,
    Artifact,
    ArtifactType,
    Evidence,
)
from ..utils.logging import get_logger

logger = get_logger("conflict")


class ConflictDetector:
    """Detects conflicts between agents and converts them to tasks"""

    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard

    def detect_conflicts(self, task_id: str) -> List[Signal]:
        """Detect conflicts for a specific task

        Conflicts occur when:
        1. Tests fail (Evidence with passed=False)
        2. Critics disagree (Claim with low confidence or negative review)
        3. Multiple incompatible artifacts exist for same task

        Args:
            task_id: Task to check for conflicts

        Returns:
            List of conflict signals
        """
        conflicts = []

        # Check 1: Test failures
        test_conflicts = self._detect_test_failures(task_id)
        conflicts.extend(test_conflicts)

        # Check 2: Critic disagreements
        critic_conflicts = self._detect_critic_disagreements(task_id)
        conflicts.extend(critic_conflicts)

        # Check 3: Artifact incompatibilities
        artifact_conflicts = self._detect_artifact_conflicts(task_id)
        conflicts.extend(artifact_conflicts)

        return conflicts

    def _detect_test_failures(self, task_id: str) -> List[Signal]:
        """Find test failure conflicts"""
        conflicts = []

        # Get latest code artifact
        latest_code = self.blackboard.get_latest_artifact(task_id, ArtifactType.CODE)
        if not latest_code:
            return conflicts

        # Check evidence for this artifact
        evidence_list = self.blackboard.get_evidence_for_artifact(latest_code.artifact_id)

        for evidence in evidence_list:
            if evidence.type == "test_result" and evidence.passed is False:
                signal = Signal(
                    type=SignalType.CONFLICT,
                    task_id=task_id,
                    description=f"Test failure: {evidence.content[:200]}",
                    severity=0.8,
                    metadata={
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": latest_code.artifact_id,
                        "conflict_type": "test_failure",
                    }
                )
                conflicts.append(signal)
                logger.info(f"Detected test failure conflict for task {task_id}")

        return conflicts

    def _detect_critic_disagreements(self, task_id: str) -> List[Signal]:
        """Find critic disagreement conflicts"""
        conflicts = []

        # Get review artifacts
        reviews = self.blackboard.get_artifacts_by_type(task_id, ArtifactType.REVIEW)

        for review in reviews:
            # Check claims associated with this review
            claims = self.blackboard.get_claims_for_artifact(review.artifact_id)

            for claim in claims:
                # Low confidence or explicit disagreement indicates conflict
                if claim.confidence < 0.5:
                    signal = Signal(
                        type=SignalType.CONFLICT,
                        task_id=task_id,
                        description=f"Critic disagreement: {claim.statement}",
                        severity=1.0 - claim.confidence,  # Lower confidence = higher severity
                        metadata={
                            "claim_id": claim.claim_id,
                            "artifact_id": review.artifact_id,
                            "conflict_type": "critic_disagreement",
                            "confidence": claim.confidence,
                        }
                    )
                    conflicts.append(signal)
                    logger.info(f"Detected critic disagreement for task {task_id}")

        return conflicts

    def _detect_artifact_conflicts(self, task_id: str) -> List[Signal]:
        """Detect incompatible artifacts (e.g., multiple code solutions)"""
        conflicts = []

        # Get all code artifacts for this task
        code_artifacts = self.blackboard.get_artifacts_by_type(task_id, ArtifactType.CODE)

        # If we have multiple code artifacts from different authors, it might be a conflict
        # (unless they're versioned updates from same author)
        if len(code_artifacts) > 1:
            authors = set(a.author for a in code_artifacts)
            if len(authors) > 1:
                signal = Signal(
                    type=SignalType.DUPLICATION,
                    task_id=task_id,
                    description=f"Multiple code solutions from {len(authors)} different agents",
                    severity=0.5,
                    metadata={
                        "conflict_type": "multiple_solutions",
                        "num_artifacts": len(code_artifacts),
                        "authors": list(authors),
                    }
                )
                conflicts.append(signal)
                logger.info(f"Detected multiple solutions conflict for task {task_id}")

        return conflicts

    def create_fix_task(self, parent_task: Task, conflict: Signal) -> Task:
        """Convert a conflict into a fix task (Feature E: Conflict-as-Task)

        Args:
            parent_task: Original task that has conflict
            conflict: Conflict signal to resolve

        Returns:
            New fix task with higher priority
        """
        # Get latest code and feedback
        latest_code = self.blackboard.get_latest_artifact(
            parent_task.task_id, ArtifactType.CODE
        )

        # Extract feedback from conflict
        feedback = conflict.description
        if "test_result" in conflict.metadata:
            # Add test output if available
            evidence_id = conflict.metadata.get("evidence_id")
            if evidence_id:
                evidence = self.blackboard.storage.get_evidence(evidence_id)
                if evidence:
                    feedback += f"\n\nTest output:\n{evidence.content}"

        # Create fix task
        fix_task = Task(
            parent_id=parent_task.task_id,
            title=f"Fix: {parent_task.title}",
            description=parent_task.description,  # Same problem
            acceptance_criteria=parent_task.acceptance_criteria,
            metadata={
                **parent_task.metadata,
                "previous_code": latest_code.content if latest_code else "",
                "feedback": feedback,
                "task_type": "code_fix",
                "conflict_id": conflict.signal_id,
                "iteration": parent_task.metadata.get("iteration", 0) + 1,
            },
            priority=parent_task.priority + 1.0,  # Higher priority
        )

        logger.info(
            f"Created fix task {fix_task.task_id} for conflict {conflict.signal_id}"
        )

        return fix_task

    def resolve_conflict(self, conflict: Signal):
        """Mark a conflict as resolved

        Args:
            conflict: Conflict signal to resolve
        """
        conflict.resolved = True
        self.blackboard.storage.update_signal(conflict)
        logger.info(f"Resolved conflict {conflict.signal_id}")
