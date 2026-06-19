"""Core Blackboard class for GAIA coordination"""

from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

from .models import (
    Task,
    TaskStatus,
    Artifact,
    ArtifactType,
    Evidence,
    Signal,
    SignalType,
    Claim,
    Policy,
    Lease,
)
from .storage import StorageBackend, InMemoryStorage
from ..utils.blackboard_logger import BlackboardLogger, EventType


class Blackboard:
    """Central coordination hub for GAIA agents

    The blackboard stores all tasks, artifacts, claims, evidence, and signals.
    Agents coordinate by reading/writing to the blackboard via this interface.
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        policy: Optional[Policy] = None,
        logger: Optional[BlackboardLogger] = None,
        log_file: Optional[Path] = None,
    ):
        self.storage = storage or InMemoryStorage()
        self.policy = policy or Policy()
        self.audit_log: List[Dict] = []

        # BlackboardLogger integration
        self.logger = logger or BlackboardLogger(
            log_file=log_file,
            log_to_console=True
        )

    # ==================== Task Management ====================

    def post_task(self, task: Task) -> str:
        """Post a new task to the blackboard"""
        self.storage.put_task(task)
        self.logger.log_task_posted(
            task_id=task.task_id,
            actor="system",
            task_type=task.metadata.get("task_type", "unknown"),
            title=task.title,
            status=task.status.value
        )
        return task.task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID"""
        return self.storage.get_task(task_id)

    def get_open_tasks(self, available_only: bool = True) -> List[Task]:
        """Get all OPEN tasks, optionally filtering to only available (unleased) ones"""
        return self.storage.query_tasks(status=TaskStatus.OPEN, available_only=available_only)

    def poll_task(self, agent_id: str, agent_tier: Optional[str] = None) -> Optional[Task]:
        """Self-assignment: agent polls for highest-priority OPEN task it can handle

        Returns the task if found, None if no suitable task available.
        Does NOT claim the task - agent must call claim_task() separately.
        """
        available_tasks = self.get_open_tasks(available_only=True)

        # Filter by routing rules if tier specified
        if agent_tier and self.policy.routing_rules:
            available_tasks = [
                t
                for t in available_tasks
                if self.policy.routing_rules.get(t.metadata.get("task_type"), agent_tier)
                == agent_tier
            ]

        # Return highest priority task
        return available_tasks[0] if available_tasks else None

    def claim_task(
        self, agent_id: str, task_id: str, lease_duration_s: Optional[int] = None
    ) -> bool:
        """Claim a task via lease mechanism

        Returns True if claim successful, False otherwise.
        """
        task = self.storage.get_task(task_id)
        if not task:
            return False

        # Check if task is available
        if not task.is_available():
            return False

        # Create lease
        duration = lease_duration_s or self.policy.task_lease_duration_seconds
        task.lease = Lease.create(agent_id, duration)
        task.status = TaskStatus.CLAIMED
        task.updated_at = datetime.utcnow()

        self.storage.update_task(task)
        self.logger.log_task_claimed(task_id=task_id, agent_id=agent_id)
        return True

    def release_task(self, task_id: str) -> None:
        """Release a task (remove lease, set back to OPEN)"""
        task = self.storage.get_task(task_id)
        if task:
            agent_id = task.lease.agent_id if task.lease else "unknown"
            task.lease = None
            task.status = TaskStatus.OPEN
            task.updated_at = datetime.utcnow()
            self.storage.update_task(task)
            self.logger.log_event(
                EventType.TASK_RELEASED,
                agent_id,
                {"task_id": task_id}
            )

    def complete_task(self, task_id: str, artifacts: Optional[List[Artifact]] = None) -> None:
        """Mark a task as completed"""
        task = self.storage.get_task(task_id)
        if task:
            agent_id = task.lease.agent_id if task.lease else "system"
            task.status = TaskStatus.DONE
            task.updated_at = datetime.utcnow()
            self.storage.update_task(task)
            self.logger.log_task_completed(
                task_id=task_id,
                agent_id=agent_id,
                artifacts_count=len(artifacts) if artifacts else 0
            )

            # Store artifacts if provided
            if artifacts:
                for artifact in artifacts:
                    self.post_artifact(artifact)

    def fail_task(self, task_id: str, reason: str) -> None:
        """Mark a task as failed"""
        task = self.storage.get_task(task_id)
        if task:
            agent_id = task.lease.agent_id if task.lease else "system"
            task.status = TaskStatus.FAILED
            task.metadata["failure_reason"] = reason
            task.updated_at = datetime.utcnow()
            self.storage.update_task(task)
            self.logger.log_task_failed(
                task_id=task_id,
                agent_id=agent_id,
                error=reason
            )

    def update_task(self, task: Task) -> None:
        """Update a task"""
        task.updated_at = datetime.utcnow()
        self.storage.update_task(task)

    # ==================== Artifact Management ====================

    def post_artifact(self, artifact: Artifact) -> str:
        """Post an artifact to the blackboard"""
        self.storage.put_artifact(artifact)
        self.logger.log_artifact_posted(
            artifact_id=artifact.artifact_id,
            actor=artifact.author,
            artifact_type=artifact.type.value,
            task_id=artifact.task_id,
            version=artifact.version
        )
        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact by ID"""
        return self.storage.get_artifact(artifact_id)

    def get_latest_artifact(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> Optional[Artifact]:
        """Get the most recent artifact for a task"""
        return self.storage.get_latest_artifact(task_id, artifact_type)

    def get_artifacts_for_task(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> List[Artifact]:
        """Get all artifacts for a task"""
        return self.storage.get_artifacts_for_task(task_id, artifact_type)

    # ==================== Evidence Management ====================

    def post_evidence(self, evidence: Evidence) -> str:
        """Post evidence to the blackboard"""
        self.storage.put_evidence(evidence)
        self.logger.log_evidence_posted(
            evidence_id=evidence.evidence_id,
            actor="system",
            evidence_type=evidence.type,
            passed=evidence.passed
        )
        return evidence.evidence_id

    def get_evidence_for_artifact(self, artifact_id: str) -> List[Evidence]:
        """Get all evidence for an artifact"""
        return self.storage.get_evidence_for_artifact(artifact_id)

    # ==================== Signal Management ====================

    def post_signal(self, signal: Signal) -> str:
        """Post a signal to the blackboard"""
        self.storage.put_signal(signal)
        self.logger.log_signal_posted(
            signal_id=signal.signal_id,
            actor="system",
            signal_type=signal.type.value,
            severity=signal.severity
        )
        return signal.signal_id

    def get_signals(
        self, resolved: Optional[bool] = None, signal_type: Optional[SignalType] = None
    ) -> List[Signal]:
        """Query signals"""
        return self.storage.get_signals(resolved=resolved, signal_type=signal_type)

    def resolve_signal(self, signal_id: str) -> None:
        """Mark a signal as resolved"""
        signal = self.storage.get_signals()  # TODO: need get by id
        for s in signal:
            if s.signal_id == signal_id:
                s.resolved = True
                self.storage.put_signal(s)
                self.logger.log_event(
                    EventType.SIGNAL_RESOLVED,
                    "system",
                    {"signal_id": signal_id}
                )
                break

    def detect_signals(self) -> List[Signal]:
        """Scan blackboard for conflicts, stale leases, etc."""
        signals = []

        # Check for failed evidence (conflict signal)
        for evidence in self.storage._evidence.values():  # type: ignore
            if evidence.passed is False:
                # Check if signal already exists for this evidence
                existing = [
                    s
                    for s in self.get_signals(resolved=False, signal_type=SignalType.CONFLICT)
                    if evidence.evidence_id in s.metadata.get("evidence_ids", [])
                ]
                if not existing:
                    signal = Signal(
                        type=SignalType.CONFLICT,
                        task_id=evidence.artifact_id or "unknown",  # Link to artifact's task
                        description=f"Test failed: {evidence.content[:100]}",
                        severity=0.8,
                        metadata={"evidence_ids": [evidence.evidence_id]},
                    )
                    self.post_signal(signal)
                    signals.append(signal)

        # Check for stale leases
        open_tasks = self.storage.query_tasks(status=TaskStatus.OPEN)
        for task in open_tasks:
            if task.lease and task.lease.is_expired():
                signal = Signal(
                    type=SignalType.STALENESS,
                    task_id=task.task_id,
                    description=f"Lease expired for task {task.title}",
                    severity=0.5,
                )
                self.post_signal(signal)
                signals.append(signal)

                # Auto-release the task
                self.release_task(task.task_id)

        return signals

    # ==================== Claim Management ====================

    def post_claim(self, claim: Claim) -> str:
        """Post a claim to the blackboard"""
        self.storage.put_claim(claim)
        self.logger.log_event(
            EventType.CLAIM_POSTED,
            claim.author,
            {
                "claim_id": claim.claim_id,
                "task_id": claim.task_id,
                "confidence": claim.confidence,
                "statement": claim.statement
            }
        )
        return claim.claim_id

    def get_claims_for_task(self, task_id: str) -> List[Claim]:
        """Get all claims for a task"""
        return self.storage.get_claims_for_task(task_id)

    # ==================== Branching (Feature F) ====================

    def fork(self, fork_id: str) -> "Blackboard":
        """Create one forked blackboard (snapshot copy) for a parallel trial"""
        snapshot = self.storage.snapshot()  # type: ignore  # InMemoryStorage specific
        branch = Blackboard(storage=InMemoryStorage(), policy=self.policy.model_copy())
        branch.storage.restore_snapshot(snapshot)  # type: ignore
        self.logger.log_event(
            EventType.BRANCH_CREATED,
            "system",
            {"fork_id": fork_id}
        )
        return branch

    def merge(self, winner: "Blackboard") -> None:
        """Merge winning fork's state back into main blackboard"""
        winner_snapshot = winner.storage.snapshot()  # type: ignore
        self.storage.restore_snapshot(winner_snapshot)  # type: ignore
        self.logger.log_event(
            EventType.BRANCH_MERGED,
            "system",
            {}
        )

    # ==================== Utility ====================

    def get_audit_log(self) -> List[Dict]:
        """Get audit log"""
        return self.audit_log

    def snapshot(self) -> Dict:
        """Create snapshot of blackboard state"""
        return {
            "storage": self.storage.snapshot(),  # type: ignore
            "policy": self.policy.model_dump(),
            "audit_log": self.audit_log.copy(),
        }

    def clear(self) -> None:
        """Clear all data (for testing)"""
        self.storage.clear()
        self.audit_log.clear()
