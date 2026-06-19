"""MiniWoB++ benchmark evaluator — computes success rates"""

from typing import List, Dict, Any
from collections import defaultdict


class MiniWoBEvaluator:
    """Evaluate MiniWoB++ episode results

    Primary metric: success_rate (fraction of tasks solved).
    Secondary: avg_steps_to_success, avg_cost_usd, per_task breakdown.
    """

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics from episode results

        Args:
            results: List of result dicts, each with:
                - task_id: str
                - task_name: str
                - passed: bool
                - steps_taken: int
                - cost_usd: float
                - duration_s: float
                - difficulty: str  (optional)

        Returns:
            Dict with:
            - success_rate: float (0.0 to 1.0)
            - total: int
            - passed: int
            - failed: int
            - avg_steps: float (on successful tasks)
            - avg_cost_usd: float
            - total_cost_usd: float
            - by_difficulty: dict (easy/medium/hard breakdown)
            - per_task: dict (task_id -> {passed, steps, cost})
        """
        if not results:
            return {"success_rate": 0.0, "total": 0, "passed": 0, "failed": 0}

        total = len(results)
        passed_results = [r for r in results if r.get("passed")]
        failed_results = [r for r in results if not r.get("passed")]

        success_rate = len(passed_results) / total if total > 0 else 0.0

        avg_steps = (
            sum(r.get("steps_taken", 0) for r in passed_results) / len(passed_results)
            if passed_results else 0.0
        )

        total_cost = sum(r.get("cost_usd", 0.0) for r in results)
        avg_cost = total_cost / total if total > 0 else 0.0

        # Breakdown by difficulty
        by_difficulty: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "passed": 0})
        for r in results:
            diff = r.get("difficulty", "unknown")
            by_difficulty[diff]["total"] += 1
            if r.get("passed"):
                by_difficulty[diff]["passed"] += 1
        for diff, stats in by_difficulty.items():
            t = stats["total"]
            stats["success_rate"] = stats["passed"] / t if t > 0 else 0.0

        # Per-task summary
        per_task = {
            r["task_id"]: {
                "passed": r.get("passed", False),
                "steps_taken": r.get("steps_taken", 0),
                "cost_usd": r.get("cost_usd", 0.0),
                "duration_s": r.get("duration_s", 0.0),
            }
            for r in results
        }

        return {
            "success_rate": success_rate,
            "total": total,
            "passed": len(passed_results),
            "failed": len(failed_results),
            "avg_steps_on_success": avg_steps,
            "avg_cost_usd": avg_cost,
            "total_cost_usd": total_cost,
            "by_difficulty": dict(by_difficulty),
            "per_task": per_task,
        }

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a human-readable summary"""
        print(f"\n{'='*60}")
        print(f"MiniWoB++ Results")
        print(f"{'='*60}")
        print(f"Success Rate: {metrics['success_rate']:.1%}  "
              f"({metrics['passed']}/{metrics['total']})")
        print(f"Avg steps (success): {metrics.get('avg_steps_on_success', 0):.1f}")
        print(f"Total cost: ${metrics.get('total_cost_usd', 0):.4f}")

        if metrics.get("by_difficulty"):
            print(f"\nBy difficulty:")
            for diff, stats in sorted(metrics["by_difficulty"].items()):
                print(f"  {diff:8s}: {stats['success_rate']:.1%}  "
                      f"({stats['passed']}/{stats['total']})")

        failed_tasks = [
            tid for tid, info in metrics.get("per_task", {}).items()
            if not info["passed"]
        ]
        if failed_tasks:
            print(f"\nFailed tasks ({len(failed_tasks)}):")
            for tid in sorted(failed_tasks):
                print(f"  - {tid}")
        print(f"{'='*60}\n")
