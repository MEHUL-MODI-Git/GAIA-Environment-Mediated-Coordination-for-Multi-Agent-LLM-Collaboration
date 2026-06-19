"""Base method interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MethodResult(BaseModel):
    """Result from solving one HumanEval problem"""

    task_id: str
    method: str
    passed: bool
    code: str = ""
    iterations: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseMethod(ABC):
    """Base class for experiment methods"""

    @abstractmethod
    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve a single HumanEval problem

        Args:
            problem: HumanEval problem dict with keys:
                - task_id: Problem ID
                - prompt: Function signature + docstring
                - test: Test harness
                - entry_point: Function name

        Returns:
            MethodResult with solution and metrics
        """
