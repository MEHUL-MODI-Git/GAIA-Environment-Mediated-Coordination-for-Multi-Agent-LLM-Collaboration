"""Core data models for GAIA blackboard"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class TaskStatus(str, Enum):
    """Status of a task in the blackboard"""

    OPEN = "OPEN"
    CLAIMED = "CLAIMED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class Lease(BaseModel):
    """Lease for task ownership"""

    agent_id: str
    claimed_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if lease has expired"""
        return datetime.utcnow() > self.expires_at

    @classmethod
    def create(cls, agent_id: str, duration_seconds: int = 120):
        """Create a new lease with specified duration"""
        now = datetime.utcnow()
        return cls(
            agent_id=agent_id, claimed_at=now, expires_at=now + timedelta(seconds=duration_seconds)
        )


class Task(BaseModel):
    """A unit of work in the blackboard"""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    title: str
    description: str
    status: TaskStatus = TaskStatus.OPEN
    deps: List[str] = Field(default_factory=list)  # task_ids this depends on
    priority: float = 1.0
    lease: Optional[Lease] = None
    acceptance_criteria: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def is_available(self) -> bool:
        """Check if task is available for claiming"""
        if self.status != TaskStatus.OPEN:
            return False
        if self.lease and not self.lease.is_expired():
            return False
        return True


class ArtifactType(str, Enum):
    """Type of artifact"""

    CODE = "CODE"
    TEST_RESULT = "TEST_RESULT"
    PLAN = "PLAN"
    REVIEW = "REVIEW"
    DOCUMENTATION = "DOCUMENTATION"


class Artifact(BaseModel):
    """A concrete output produced by an agent"""

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ArtifactType
    version: int = 1
    task_id: str
    author: str  # agent_id
    content: str  # the actual code, plan text, etc.
    provenance: List[str] = Field(default_factory=list)  # artifact_ids this derives from
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A statement asserted by an agent"""

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    evidence_ids: List[str] = Field(default_factory=list)
    task_id: str
    author: str  # agent_id
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Evidence(BaseModel):
    """Objective support for claims"""

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "test_result", "log", "calculation", etc.
    content: str  # test output, pass/fail, traceback, etc.
    artifact_id: Optional[str] = None
    passed: Optional[bool] = None  # for test evidence
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignalType(str, Enum):
    """System signals/flags"""

    CONFLICT = "CONFLICT"
    UNCERTAINTY = "UNCERTAINTY"
    DUPLICATION = "DUPLICATION"
    URGENCY = "URGENCY"
    STALENESS = "STALENESS"
    BUDGET_RISK = "BUDGET_RISK"


class Signal(BaseModel):
    """System indicator/flag"""

    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: SignalType
    task_id: str
    description: str
    severity: float = Field(default=0.5, ge=0.0, le=1.0)  # 0.0 to 1.0
    resolved: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Policy(BaseModel):
    """Coordination policies and thresholds"""

    # Routing
    routing_rules: Dict[str, str] = Field(default_factory=dict)  # task_type -> agent_tier

    # Spawning (Feature D)
    spawn_threshold: int = 3  # spawn new agent if OPEN backlog exceeds this
    max_agents: int = 10  # max total agents

    # Branching (Feature F)
    branch_trigger_on_failure: bool = False
    branch_trigger_on_uncertainty: bool = False
    branch_max_parallel: int = 3
    uncertainty_threshold: float = 0.7  # trigger branch if uncertainty > this

    # Verification
    verification_strictness: str = "all_tests_pass"
    max_retries: int = 3
    stop_on_first_pass: bool = True

    # Meta-update (Feature G)
    meta_update_enabled: bool = False
    meta_update_interval: int = 10  # update every N episodes

    # Stop conditions
    max_iterations: int = 10
    max_cost_usd: float = 10.0
    timeout_seconds: int = 300

    # Lease duration
    task_lease_duration_seconds: int = 120
