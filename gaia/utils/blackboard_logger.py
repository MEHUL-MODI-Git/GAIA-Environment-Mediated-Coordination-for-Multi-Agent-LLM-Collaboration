"""Comprehensive blackboard event logging system

Tracks EVERY change to the blackboard:
- Task creations, status changes, claims, completions
- Artifact creations, versions
- Evidence posted
- Signals detected
- Claims made
- Metrics: tokens, cost, time for every operation
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class EventType(str, Enum):
    """Types of blackboard events"""
    # Task events
    TASK_POSTED = "task_posted"
    TASK_CLAIMED = "task_claimed"
    TASK_RELEASED = "task_released"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_STATUS_CHANGE = "task_status_change"

    # Artifact events
    ARTIFACT_POSTED = "artifact_posted"
    ARTIFACT_UPDATED = "artifact_updated"

    # Evidence events
    EVIDENCE_POSTED = "evidence_posted"

    # Signal events
    SIGNAL_POSTED = "signal_posted"
    SIGNAL_RESOLVED = "signal_resolved"

    # Claim events
    CLAIM_POSTED = "claim_posted"

    # Agent events
    AGENT_POLL = "agent_poll"
    AGENT_EXECUTE_START = "agent_execute_start"
    AGENT_EXECUTE_END = "agent_execute_end"
    LLM_CALL = "llm_call"

    # Episode events
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"

    # Resolution events
    CONFLICT_DETECTED = "conflict_detected"
    BRANCH_CREATED = "branch_created"
    BRANCH_MERGED = "branch_merged"


class BlackboardLogger:
    """Comprehensive event logger for blackboard operations

    Logs EVERYTHING that happens:
    - Who did what
    - When it happened
    - What changed
    - Metrics (tokens, cost, time)
    """

    def __init__(self, log_file: Optional[Path] = None, log_to_console: bool = True):
        self.log_file = log_file
        self.log_to_console = log_to_console
        self.events = []  # In-memory event log
        self.start_time = time.time()

        # Per-episode metrics
        self.episode_metrics = {
            "total_events": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "total_llm_calls": 0,
            "tasks_created": 0,
            "artifacts_created": 0,
            "conflicts_detected": 0,
        }

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: EventType,
        actor: str,  # Which agent/system
        details: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log a blackboard event

        Args:
            event_type: Type of event
            actor: Who triggered this event (agent_id, "system", "episode_loop")
            details: Event-specific details
            metrics: Optional metrics (tokens, cost, time)
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            "event_type": event_type.value,
            "actor": actor,
            "details": details,
            "metrics": metrics or {},
        }

        # Add to in-memory log
        self.events.append(event)
        self.episode_metrics["total_events"] += 1

        # Update metrics
        if metrics:
            if "cost_usd" in metrics:
                self.episode_metrics["total_cost_usd"] += metrics["cost_usd"]
            if "tokens" in metrics:
                self.episode_metrics["total_tokens"] += metrics["tokens"]
            if event_type == EventType.LLM_CALL:
                self.episode_metrics["total_llm_calls"] += 1

        # Write to file immediately (append mode)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")

        # Console logging (optional)
        if self.log_to_console:
            self._print_event(event)

    def _print_event(self, event: dict):
        """Pretty print event to console"""
        elapsed = event["elapsed_seconds"]
        event_type = event["event_type"]
        actor = event["actor"]

        # Color codes
        colors = {
            "task": "\033[94m",  # Blue
            "artifact": "\033[92m",  # Green
            "signal": "\033[91m",  # Red
            "llm": "\033[95m",  # Magenta
            "episode": "\033[96m",  # Cyan
            "reset": "\033[0m",
        }

        # Pick color based on event type
        if "task" in event_type:
            color = colors["task"]
        elif "artifact" in event_type:
            color = colors["artifact"]
        elif "signal" in event_type or "conflict" in event_type:
            color = colors["signal"]
        elif "llm" in event_type:
            color = colors["llm"]
        elif "episode" in event_type or "iteration" in event_type:
            color = colors["episode"]
        else:
            color = ""

        # Format message
        msg = f"[{elapsed:6.2f}s] {color}{event_type:20s}{colors['reset']} | {actor:15s}"

        # Add key details
        details = event.get("details", {})
        if "task_id" in details:
            msg += f" | task={details['task_id'][:8]}"
        if "artifact_id" in details:
            msg += f" | artifact={details['artifact_id'][:8]}"

        # Add metrics if present
        metrics = event.get("metrics", {})
        if "cost_usd" in metrics:
            msg += f" | ${metrics['cost_usd']:.4f}"
        if "tokens" in metrics:
            msg += f" | {metrics['tokens']}tok"
        if "duration_ms" in metrics:
            msg += f" | {metrics['duration_ms']:.0f}ms"

        print(msg)

    def log_task_posted(self, task_id: str, actor: str, task_type: str, **kwargs):
        """Log task creation"""
        self.episode_metrics["tasks_created"] += 1
        self.log_event(
            EventType.TASK_POSTED,
            actor,
            {
                "task_id": task_id,
                "task_type": task_type,
                **kwargs
            }
        )

    def log_task_claimed(self, task_id: str, agent_id: str):
        """Log task claim"""
        self.log_event(
            EventType.TASK_CLAIMED,
            agent_id,
            {"task_id": task_id}
        )

    def log_task_completed(self, task_id: str, agent_id: str, artifacts_count: int):
        """Log task completion"""
        self.log_event(
            EventType.TASK_COMPLETED,
            agent_id,
            {
                "task_id": task_id,
                "artifacts_produced": artifacts_count
            }
        )

    def log_task_failed(self, task_id: str, agent_id: str, error: str):
        """Log task failure"""
        self.log_event(
            EventType.TASK_FAILED,
            agent_id,
            {
                "task_id": task_id,
                "error": error
            }
        )

    def log_artifact_posted(
        self, artifact_id: str, actor: str, artifact_type: str, task_id: str, version: int
    ):
        """Log artifact creation"""
        self.episode_metrics["artifacts_created"] += 1
        self.log_event(
            EventType.ARTIFACT_POSTED,
            actor,
            {
                "artifact_id": artifact_id,
                "type": artifact_type,
                "task_id": task_id,
                "version": version
            }
        )

    def log_evidence_posted(
        self, evidence_id: str, actor: str, evidence_type: str, passed: Optional[bool]
    ):
        """Log evidence creation (test results)"""
        self.log_event(
            EventType.EVIDENCE_POSTED,
            actor,
            {
                "evidence_id": evidence_id,
                "type": evidence_type,
                "passed": passed
            }
        )

    def log_signal_posted(self, signal_id: str, actor: str, signal_type: str, severity: float):
        """Log signal detection"""
        if "conflict" in signal_type.lower():
            self.episode_metrics["conflicts_detected"] += 1
        self.log_event(
            EventType.SIGNAL_POSTED,
            actor,
            {
                "signal_id": signal_id,
                "type": signal_type,
                "severity": severity
            }
        )

    def log_llm_call(
        self,
        agent_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ):
        """Log LLM API call with full metrics"""
        total_tokens = prompt_tokens + completion_tokens
        self.log_event(
            EventType.LLM_CALL,
            agent_id,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            metrics={
                "cost_usd": cost_usd,
                "tokens": total_tokens,
                "duration_ms": latency_ms,
            }
        )

    def log_agent_execute(
        self, agent_id: str, task_id: str, start: bool = True, duration_ms: float = None
    ):
        """Log agent execution start/end"""
        event_type = EventType.AGENT_EXECUTE_START if start else EventType.AGENT_EXECUTE_END
        details = {"task_id": task_id}
        metrics = {"duration_ms": duration_ms} if duration_ms else None

        self.log_event(event_type, agent_id, details, metrics)

    def log_episode_start(self, problem_id: str):
        """Log episode start"""
        self.start_time = time.time()
        self.log_event(
            EventType.EPISODE_START,
            "episode_loop",
            {"problem_id": problem_id}
        )

    def log_episode_end(self, problem_id: str, passed: bool, code: str = ""):
        """Log episode end"""
        duration = time.time() - self.start_time
        self.log_event(
            EventType.EPISODE_END,
            "episode_loop",
            {
                "problem_id": problem_id,
                "passed": passed,
                "code_length": len(code)
            },
            metrics={"duration_ms": duration * 1000}
        )

    def log_iteration(self, iteration_num: int, start: bool = True):
        """Log iteration start/end"""
        event_type = EventType.ITERATION_START if start else EventType.ITERATION_END
        self.log_event(
            event_type,
            "episode_loop",
            {"iteration": iteration_num}
        )

    def log_conflict_detected(self, task_id: str, conflict_type: str, failure_count: int):
        """Log conflict detection"""
        self.log_event(
            EventType.CONFLICT_DETECTED,
            "episode_loop",
            {
                "task_id": task_id,
                "conflict_type": conflict_type,
                "failure_count": failure_count
            }
        )

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of episode metrics"""
        duration = time.time() - self.start_time
        return {
            **self.episode_metrics,
            "duration_seconds": duration,
            "total_events": len(self.events),
            "events_per_second": len(self.events) / duration if duration > 0 else 0,
        }

    def save_summary(self, output_path: Path):
        """Save episode summary to JSON"""
        summary = self.get_episode_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def get_event_timeline(self) -> list:
        """Get chronological event timeline"""
        return sorted(self.events, key=lambda e: e["timestamp"])

    def get_events_by_actor(self, actor: str) -> list:
        """Get all events by specific actor"""
        return [e for e in self.events if e["actor"] == actor]

    def get_events_by_type(self, event_type: EventType) -> list:
        """Get all events of specific type"""
        return [e for e in self.events if e["event_type"] == event_type.value]

    def reset(self):
        """Reset for new episode"""
        self.events = []
        self.start_time = time.time()
        self.episode_metrics = {
            "total_events": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "total_llm_calls": 0,
            "tasks_created": 0,
            "artifacts_created": 0,
            "conflicts_detected": 0,
        }
