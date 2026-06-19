"""Checkpoint manager for long-running experiments"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """Manages checkpointing for multi-problem runs

    Provides crash recovery, progress tracking, and resumption.
    """

    def __init__(self, checkpoint_path: Path):
        """
        Args:
            checkpoint_path: Path to checkpoint file (e.g., results/run.checkpoint.json)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data = {
            "run_started_at": None,
            "last_updated_at": None,
            "total_problems": 0,
            "completed_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "error_count": 0,
            "total_cost_usd": 0.0,
            "total_duration_s": 0.0,
            "results": [],  # List of {task_id, passed, iterations, cost, duration, error, ...}
        }
        self._load()

    def _load(self):
        """Load existing checkpoint if available"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as f:
                    self.data = json.load(f)
                print(f"✓ Loaded checkpoint: {self.completed_count}/{self.total_problems} completed")
            except Exception as e:
                print(f"⚠ Checkpoint corrupted, starting fresh: {e}")
                self.checkpoint_path.unlink(missing_ok=True)

    def _save(self):
        """Save checkpoint to disk"""
        self.data["last_updated_at"] = datetime.now().isoformat()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        temp_path.replace(self.checkpoint_path)

    def start_run(self, total_problems: int):
        """Initialize a new run (or resume existing)"""
        if self.data["run_started_at"] is None:
            self.data["run_started_at"] = datetime.now().isoformat()
        self.data["total_problems"] = total_problems
        self._save()

    def is_completed(self, task_id: str) -> bool:
        """Check if a problem is already completed"""
        return any(r["task_id"] == task_id for r in self.data["results"])

    def add_result(
        self,
        task_id: str,
        passed: bool,
        iterations: int,
        cost_usd: float,
        duration_s: float,
        stop_reason: str,
        num_conflicts: int = 0,
        num_branches: int = 0,
        error: Optional[str] = None,
    ):
        """Record a problem result and update checkpoint"""
        result = {
            "task_id": task_id,
            "passed": passed,
            "iterations": iterations,
            "cost_usd": cost_usd,
            "duration_s": duration_s,
            "stop_reason": stop_reason,
            "num_conflicts": num_conflicts,
            "num_branches": num_branches,
            "error": error,
            "completed_at": datetime.now().isoformat(),
        }

        self.data["results"].append(result)
        self.data["completed_count"] += 1
        self.data["total_cost_usd"] += cost_usd
        self.data["total_duration_s"] += duration_s

        if error:
            self.data["error_count"] += 1
        elif passed:
            self.data["passed_count"] += 1
        else:
            self.data["failed_count"] += 1

        self._save()

    def finalize(self, output_path: Path):
        """Save final results and cleanup checkpoint"""
        # Save final results
        final_data = {
            "run_started_at": self.data["run_started_at"],
            "run_completed_at": datetime.now().isoformat(),
            "total_problems": self.total_problems,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "errors": self.error_count,
            "pass_rate": self.passed_count / max(self.completed_count, 1),
            "total_cost_usd": self.total_cost_usd,
            "total_duration_s": self.total_duration_s,
            "average_duration_s": self.total_duration_s / max(self.completed_count, 1),
            "results": self.data["results"],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        # Delete checkpoint
        self.checkpoint_path.unlink(missing_ok=True)

        return final_data

    @property
    def completed_count(self) -> int:
        return self.data["completed_count"]

    @property
    def passed_count(self) -> int:
        return self.data["passed_count"]

    @property
    def failed_count(self) -> int:
        return self.data["failed_count"]

    @property
    def error_count(self) -> int:
        return self.data["error_count"]

    @property
    def total_problems(self) -> int:
        return self.data["total_problems"]

    @property
    def total_cost_usd(self) -> float:
        return self.data["total_cost_usd"]

    @property
    def total_duration_s(self) -> float:
        return self.data["total_duration_s"]

    @property
    def pass_rate(self) -> float:
        if self.completed_count == 0:
            return 0.0
        return self.passed_count / self.completed_count

    def get_completed_task_ids(self) -> List[str]:
        """Get list of completed task IDs"""
        return [r["task_id"] for r in self.data["results"]]
