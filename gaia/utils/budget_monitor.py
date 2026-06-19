"""Budget monitoring and cost failsafes"""

from typing import Optional
from ..utils.logging import get_logger

logger = get_logger("budget")


class BudgetMonitor:
    """Monitor and enforce cost limits during episodes

    Prevents runaway costs by tracking spend per problem
    and stopping execution when limits are exceeded.
    """

    def __init__(
        self,
        max_cost_per_problem: float = 0.30,  # $0.30 default
        max_iterations: int = 15,
        max_llm_calls: int = 50,
        warn_threshold: float = 0.75,  # Warn at 75%
    ):
        self.max_cost_per_problem = max_cost_per_problem
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.warn_threshold = warn_threshold

        # Per-problem tracking
        self.current_cost = 0.0
        self.current_iterations = 0
        self.current_llm_calls = 0

    def record_llm_call(self, cost_usd: float) -> bool:
        """Record an LLM call and check if we should continue

        Args:
            cost_usd: Cost of this LLM call

        Returns:
            True if OK to continue, False if budget exceeded
        """
        self.current_cost += cost_usd
        self.current_llm_calls += 1

        # Check cost limit
        if self.current_cost > self.max_cost_per_problem:
            logger.error(
                f"💰 BUDGET EXCEEDED: ${self.current_cost:.4f} > "
                f"${self.max_cost_per_problem} limit"
            )
            return False

        # Warn at threshold
        if self.current_cost > self.max_cost_per_problem * self.warn_threshold:
            pct = (self.current_cost / self.max_cost_per_problem) * 100
            logger.warning(
                f"⚠️  Budget at {pct:.0f}%: ${self.current_cost:.4f} / "
                f"${self.max_cost_per_problem}"
            )

        # Check LLM call limit
        if self.current_llm_calls > self.max_llm_calls:
            logger.error(
                f"🚫 LLM CALL LIMIT: {self.current_llm_calls} > "
                f"{self.max_llm_calls} max calls"
            )
            return False

        return True

    def record_iteration(self) -> bool:
        """Record an iteration and check if we should continue

        Returns:
            True if OK to continue, False if iteration limit exceeded
        """
        self.current_iterations += 1

        if self.current_iterations > self.max_iterations:
            logger.error(
                f"🔄 ITERATION LIMIT: {self.current_iterations} > "
                f"{self.max_iterations} max iterations"
            )
            return False

        return True

    def should_continue(self) -> tuple[bool, Optional[str]]:
        """Check if episode should continue

        Returns:
            Tuple of (should_continue, reason_if_stopped)
        """
        if self.current_cost > self.max_cost_per_problem:
            return False, f"Budget exceeded: ${self.current_cost:.4f}"

        if self.current_iterations > self.max_iterations:
            return False, f"Iteration limit: {self.current_iterations}"

        if self.current_llm_calls > self.max_llm_calls:
            return False, f"LLM call limit: {self.current_llm_calls}"

        return True, None

    def get_summary(self) -> dict:
        """Get budget usage summary"""
        return {
            "cost_usd": self.current_cost,
            "cost_limit_usd": self.max_cost_per_problem,
            "cost_pct": (self.current_cost / self.max_cost_per_problem) * 100,
            "iterations": self.current_iterations,
            "iteration_limit": self.max_iterations,
            "llm_calls": self.current_llm_calls,
            "llm_call_limit": self.max_llm_calls,
        }

    def reset(self):
        """Reset for next problem"""
        self.current_cost = 0.0
        self.current_iterations = 0
        self.current_llm_calls = 0
        logger.info("Budget monitor reset for next problem")


class BudgetExceededException(Exception):
    """Raised when budget limits are exceeded"""
    pass
