"""Metrics collection and tracking"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json


class LLMCallMetrics(BaseModel):
    """Metrics for a single LLM call"""

    agent_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EpisodeMetrics(BaseModel):
    """Metrics for a single episode (one HumanEval problem)"""

    task_id: str
    method: str
    passed: bool
    iterations: int
    llm_calls: List[LLMCallMetrics] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    conflicts_detected: int = 0
    branches_created: int = 0
    merge_success: Optional[bool] = None
    final_code: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Collect and aggregate metrics across episodes"""

    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None

    def start_episode(self, task_id: str, method: str):
        """Start tracking a new episode"""
        self.current_episode = EpisodeMetrics(
            task_id=task_id, method=method, passed=False, iterations=0
        )

    def record_llm_call(
        self,
        agent_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ):
        """Record a single LLM call"""
        if not self.current_episode:
            return

        call = LLMCallMetrics(
            agent_id=agent_id,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        self.current_episode.llm_calls.append(call)
        self.current_episode.total_tokens += call.total_tokens
        self.current_episode.total_cost_usd += call.cost_usd
        self.current_episode.total_latency_ms += call.latency_ms

    def record_iteration(self):
        """Record an iteration"""
        if self.current_episode:
            self.current_episode.iterations += 1

    def record_conflict(self):
        """Record a conflict detection"""
        if self.current_episode:
            self.current_episode.conflicts_detected += 1

    def record_branch(self, n_branches: int):
        """Record branch creation"""
        if self.current_episode:
            self.current_episode.branches_created = n_branches

    def record_merge(self, success: bool):
        """Record merge result"""
        if self.current_episode:
            self.current_episode.merge_success = success

    def end_episode(self, passed: bool, final_code: str = ""):
        """Finish tracking current episode"""
        if self.current_episode:
            self.current_episode.passed = passed
            self.current_episode.final_code = final_code
            self.episodes.append(self.current_episode)
            self.current_episode = None

    # ==================== Aggregate Metrics ====================

    def compute_pass_at_1(self) -> float:
        """Compute pass@1 rate"""
        if not self.episodes:
            return 0.0
        passed = sum(1 for e in self.episodes if e.passed)
        return passed / len(self.episodes)

    def compute_pass_at_k(self, k: int) -> float:
        """Compute pass@k rate (simplified - assumes independent trials)"""
        # For now, same as pass@1 since we don't have multiple samples per task
        return self.compute_pass_at_1()

    def total_cost(self) -> float:
        """Total cost across all episodes"""
        return sum(e.total_cost_usd for e in self.episodes)

    def total_tokens(self) -> int:
        """Total tokens across all episodes"""
        return sum(e.total_tokens for e in self.episodes)

    def average_latency(self) -> float:
        """Average latency per episode"""
        if not self.episodes:
            return 0.0
        return sum(e.total_latency_ms for e in self.episodes) / len(self.episodes)

    def average_iterations(self) -> float:
        """Average iterations per episode"""
        if not self.episodes:
            return 0.0
        return sum(e.iterations for e in self.episodes) / len(self.episodes)

    def conflict_rate(self) -> float:
        """Average conflicts per episode"""
        if not self.episodes:
            return 0.0
        return sum(e.conflicts_detected for e in self.episodes) / len(self.episodes)

    def branch_frequency(self) -> float:
        """Fraction of episodes that used branching"""
        if not self.episodes:
            return 0.0
        branched = sum(1 for e in self.episodes if e.branches_created > 0)
        return branched / len(self.episodes)

    # ==================== Export ====================

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dict"""
        return {
            "total_episodes": len(self.episodes),
            "pass_at_1": self.compute_pass_at_1(),
            "total_cost_usd": self.total_cost(),
            "total_tokens": self.total_tokens(),
            "average_latency_ms": self.average_latency(),
            "average_iterations": self.average_iterations(),
            "conflict_rate": self.conflict_rate(),
            "branch_frequency": self.branch_frequency(),
            "episodes": [e.model_dump() for e in self.episodes],
        }

    def to_jsonl(self, path: str):
        """Write episode results to JSONL file"""
        with open(path, "w") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode.model_dump(), default=str) + "\n")

    def summary(self) -> str:
        """Generate summary report"""
        return f"""
=== GAIA Metrics Summary ===
Episodes: {len(self.episodes)}
Pass@1: {self.compute_pass_at_1():.3f}
Total Cost: ${self.total_cost():.2f}
Total Tokens: {self.total_tokens():,}
Avg Latency: {self.average_latency():.0f}ms
Avg Iterations: {self.average_iterations():.1f}
Conflict Rate: {self.conflict_rate():.2f}
Branch Frequency: {self.branch_frequency():.2f}
"""
