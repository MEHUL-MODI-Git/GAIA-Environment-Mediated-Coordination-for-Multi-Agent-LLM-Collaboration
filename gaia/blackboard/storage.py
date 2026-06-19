"""Storage backends for blackboard"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .models import Task, Artifact, Evidence, Signal, Claim, TaskStatus, ArtifactType, SignalType


class StorageBackend(ABC):
    """Abstract storage backend interface"""

    @abstractmethod
    def put_task(self, task: Task) -> None:
        """Store a task"""

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID"""

    @abstractmethod
    def query_tasks(
        self,
        status: Optional[TaskStatus] = None,
        parent_id: Optional[str] = None,
        available_only: bool = False,
    ) -> List[Task]:
        """Query tasks with filters"""

    @abstractmethod
    def update_task(self, task: Task) -> None:
        """Update an existing task"""

    @abstractmethod
    def put_artifact(self, artifact: Artifact) -> None:
        """Store an artifact"""

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID"""

    @abstractmethod
    def get_artifacts_for_task(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> List[Artifact]:
        """Get all artifacts for a task, optionally filtered by type"""

    @abstractmethod
    def get_latest_artifact(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> Optional[Artifact]:
        """Get the most recent artifact for a task"""

    @abstractmethod
    def put_evidence(self, evidence: Evidence) -> None:
        """Store evidence"""

    @abstractmethod
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve evidence by ID"""

    @abstractmethod
    def get_evidence_for_artifact(self, artifact_id: str) -> List[Evidence]:
        """Get all evidence for an artifact"""

    @abstractmethod
    def put_signal(self, signal: Signal) -> None:
        """Store a signal"""

    @abstractmethod
    def get_signals(
        self, resolved: Optional[bool] = None, signal_type: Optional[SignalType] = None
    ) -> List[Signal]:
        """Query signals"""

    @abstractmethod
    def put_claim(self, claim: Claim) -> None:
        """Store a claim"""

    @abstractmethod
    def get_claims_for_task(self, task_id: str) -> List[Claim]:
        """Get all claims for a task"""

    @abstractmethod
    def clear(self) -> None:
        """Clear all data (for testing)"""


class InMemoryStorage(StorageBackend):
    """In-memory storage implementation using dicts"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._artifacts: Dict[str, Artifact] = {}
        self._artifact_insertion_order: List[str] = []  # artifact_ids in insertion order
        self._evidence: Dict[str, Evidence] = {}
        self._signals: Dict[str, Signal] = {}
        self._claims: Dict[str, Claim] = {}

    def put_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def query_tasks(
        self,
        status: Optional[TaskStatus] = None,
        parent_id: Optional[str] = None,
        available_only: bool = False,
    ) -> List[Task]:
        tasks = list(self._tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if parent_id is not None:
            tasks = [t for t in tasks if t.parent_id == parent_id]

        if available_only:
            tasks = [t for t in tasks if t.is_available()]

        # Sort by priority (descending) then created_at (ascending)
        tasks.sort(key=lambda t: (-t.priority, t.created_at))
        return tasks

    def update_task(self, task: Task) -> None:
        if task.task_id not in self._tasks:
            raise ValueError(f"Task {task.task_id} not found")
        self._tasks[task.task_id] = task

    def put_artifact(self, artifact: Artifact) -> None:
        self._artifacts[artifact.artifact_id] = artifact
        self._artifact_insertion_order.append(artifact.artifact_id)

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        return self._artifacts.get(artifact_id)

    def get_artifacts_for_task(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> List[Artifact]:
        # Return in insertion order (oldest first)
        artifacts = [
            self._artifacts[aid]
            for aid in self._artifact_insertion_order
            if aid in self._artifacts and self._artifacts[aid].task_id == task_id
        ]

        if artifact_type is not None:
            artifacts = [a for a in artifacts if a.type == artifact_type]

        # Sort by version descending (stable, so equal versions keep insertion order)
        artifacts.sort(key=lambda a: -a.version)
        return artifacts

    def get_latest_artifact(
        self, task_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> Optional[Artifact]:
        # Walk insertion order in REVERSE — return the most recently posted matching artifact
        for aid in reversed(self._artifact_insertion_order):
            a = self._artifacts.get(aid)
            if a is None or a.task_id != task_id:
                continue
            if artifact_type is not None and a.type != artifact_type:
                continue
            return a
        return None

    def put_evidence(self, evidence: Evidence) -> None:
        self._evidence[evidence.evidence_id] = evidence

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self._evidence.get(evidence_id)

    def get_evidence_for_artifact(self, artifact_id: str) -> List[Evidence]:
        return [e for e in self._evidence.values() if e.artifact_id == artifact_id]

    def put_signal(self, signal: Signal) -> None:
        self._signals[signal.signal_id] = signal

    def get_signals(
        self, resolved: Optional[bool] = None, signal_type: Optional[SignalType] = None
    ) -> List[Signal]:
        signals = list(self._signals.values())

        if resolved is not None:
            signals = [s for s in signals if s.resolved == resolved]

        if signal_type is not None:
            signals = [s for s in signals if s.type == signal_type]

        # Sort by severity (descending) then created_at (ascending)
        signals.sort(key=lambda s: (-s.severity, s.created_at))
        return signals

    def put_claim(self, claim: Claim) -> None:
        self._claims[claim.claim_id] = claim

    def get_claims_for_task(self, task_id: str) -> List[Claim]:
        return [c for c in self._claims.values() if c.task_id == task_id]

    def clear(self) -> None:
        """Clear all data"""
        self._tasks.clear()
        self._artifacts.clear()
        self._evidence.clear()
        self._signals.clear()
        self._claims.clear()

    def snapshot(self) -> Dict:
        """Create a snapshot of current state (for branching)"""
        return {
            "tasks": {k: v.model_copy(deep=True) for k, v in self._tasks.items()},
            "artifacts": {k: v.model_copy(deep=True) for k, v in self._artifacts.items()},
            "evidence": {k: v.model_copy(deep=True) for k, v in self._evidence.items()},
            "signals": {k: v.model_copy(deep=True) for k, v in self._signals.items()},
            "claims": {k: v.model_copy(deep=True) for k, v in self._claims.items()},
        }

    def restore_snapshot(self, snapshot: Dict) -> None:
        """Restore from a snapshot"""
        self._tasks = snapshot["tasks"]
        self._artifacts = snapshot["artifacts"]
        self._evidence = snapshot["evidence"]
        self._signals = snapshot["signals"]
        self._claims = snapshot["claims"]
