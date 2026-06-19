"""HumanEval evaluation metrics"""

from typing import List, Dict
import math
from collections import Counter


class HumanEvalEvaluator:
    """Compute HumanEval metrics (pass@1, pass@k)"""

    @staticmethod
    def compute_pass_at_1(results: List[Dict]) -> float:
        """Compute pass@1 metric

        Args:
            results: List of result dicts with 'passed' field

        Returns:
            Pass@1 rate (fraction of problems solved)
        """
        if not results:
            return 0.0

        passed_count = sum(1 for r in results if r.get("passed", False))
        return passed_count / len(results)

    @staticmethod
    def compute_pass_at_k(results: List[Dict], k: int) -> float:
        """Compute pass@k metric

        pass@k := E[1 - comb(n-c, k) / comb(n, k)]
        where n = total samples per problem, c = correct samples

        For single-sample case (k=1), this reduces to pass@1.

        Args:
            results: List of result dicts grouped by task_id
            k: Number of samples to consider

        Returns:
            Pass@k estimate
        """
        # Group results by task_id
        task_results = {}
        for r in results:
            task_id = r.get("task_id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(r.get("passed", False))

        # Compute pass@k for each task
        pass_at_k_values = []
        for task_id, passed_list in task_results.items():
            n = len(passed_list)
            c = sum(passed_list)

            if n < k:
                # Not enough samples
                continue

            # Compute pass@k for this task
            if c >= k:
                # If we have k or more correct samples, pass@k = 1
                pass_at_k_values.append(1.0)
            elif c == 0:
                # If no correct samples, pass@k = 0
                pass_at_k_values.append(0.0)
            else:
                # Use the formula
                pass_at_k = 1.0 - HumanEvalEvaluator._comb(n - c, k) / HumanEvalEvaluator._comb(n, k)
                pass_at_k_values.append(pass_at_k)

        return sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

    @staticmethod
    def _comb(n: int, k: int) -> float:
        """Compute binomial coefficient n choose k"""
        if k > n or k < 0:
            return 0.0
        if k == 0 or k == n:
            return 1.0

        k = min(k, n - k)  # Optimization
        result = 1.0
        for i in range(k):
            result *= (n - i) / (i + 1)
        return result

    @staticmethod
    def compute_metrics_summary(results: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive metrics summary

        Args:
            results: List of result dicts

        Returns:
            Dict with all metrics
        """
        if not results:
            return {
                "total_problems": 0,
                "pass_at_1": 0.0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
            }

        return {
            "total_problems": len(results),
            "pass_at_1": HumanEvalEvaluator.compute_pass_at_1(results),
            "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in results),
            "avg_latency_ms": sum(r.get("latency_ms", 0.0) for r in results) / len(results),
            "avg_iterations": sum(r.get("iterations", 0) for r in results) / len(results),
        }
