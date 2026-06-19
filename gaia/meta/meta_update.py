"""Meta-update mechanism (Feature G: Cross-Episode Policy Tuning)"""

from typing import List, Dict, Any, Optional
import statistics

from ..episode.loop import EpisodeResult
from .policy import PolicyManager
from ..utils.logging import get_logger

logger = get_logger("meta_update")


class MetaUpdater:
    """Learns from episode outcomes to tune policy parameters

    Feature G: After running multiple episodes, analyze metrics and
    adjust policy settings to improve performance.

    Tunable parameters:
    - spawn_threshold: When to create more agents
    - branch_trigger_on_failure: When to use branch-and-merge
    - max_retries: How many attempts before giving up
    - verification_strictness: How strict to be with acceptance
    """

    def __init__(self, policy_manager: PolicyManager, update_frequency: int = 10):
        self.policy_manager = policy_manager
        self.update_frequency = update_frequency  # Episodes between updates
        self.episode_history: List[EpisodeResult] = []
        self.update_count = 0

    def record_episode(self, result: EpisodeResult):
        """Record an episode result for meta-learning

        Args:
            result: Outcome of an episode
        """
        self.episode_history.append(result)
        logger.info(
            f"Recorded episode {result.task_id}: "
            f"passed={result.passed}, iterations={result.iterations}"
        )

        # Check if it's time to update
        if len(self.episode_history) % self.update_frequency == 0:
            self.update_policy()

    def update_policy(self):
        """Analyze recent episodes and update policy"""
        if not self.episode_history:
            logger.warning("No episode history to learn from")
            return

        logger.info(f"Running meta-update on {len(self.episode_history)} episodes")

        # Compute metrics from recent episodes
        recent_window = self.episode_history[-self.update_frequency:]
        metrics = self._compute_metrics(recent_window)

        # Make policy updates based on metrics
        updates = self._decide_updates(metrics)

        if updates:
            self.policy_manager.update_policy(updates)
            self.update_count += 1
            logger.info(f"Meta-update #{self.update_count} applied: {updates}")
        else:
            logger.info("No policy updates needed")

    def _compute_metrics(self, episodes: List[EpisodeResult]) -> Dict[str, Any]:
        """Compute aggregate metrics from episode results

        Args:
            episodes: List of episode results

        Returns:
            Dictionary of metrics
        """
        if not episodes:
            return {}

        pass_rate = sum(1 for e in episodes if e.passed) / len(episodes)
        avg_iterations = statistics.mean(e.iterations for e in episodes)

        # Count episodes with conflicts
        conflict_rate = sum(
            1 for e in episodes if e.conflicts_detected > 0
        ) / len(episodes)

        # Count episodes where branching was used
        branch_rate = sum(
            1 for e in episodes if e.branches_created > 0
        ) / len(episodes)

        metrics = {
            "pass_rate": pass_rate,
            "avg_iterations": avg_iterations,
            "conflict_rate": conflict_rate,
            "branch_rate": branch_rate,
            "n_episodes": len(episodes),
        }

        logger.info(f"Computed metrics: {metrics}")
        return metrics

    def _decide_updates(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Decide policy updates based on metrics

        Simple heuristics:
        - If pass_rate is low and avg_iterations is high -> increase max_retries
        - If conflict_rate is high and branch_rate is low -> enable branching
        - If pass_rate is high -> maybe reduce max_retries to save cost

        Args:
            metrics: Computed metrics

        Returns:
            Dictionary of policy updates
        """
        updates = {}
        policy = self.policy_manager.get_policy()

        pass_rate = metrics.get("pass_rate", 0)
        avg_iterations = metrics.get("avg_iterations", 0)
        conflict_rate = metrics.get("conflict_rate", 0)
        branch_rate = metrics.get("branch_rate", 0)

        # Rule 1: Low pass rate + high iterations -> increase retries
        if pass_rate < 0.5 and avg_iterations > policy.max_iterations * 0.8:
            new_max = min(policy.max_iterations + 2, 20)  # Cap at 20
            if new_max > policy.max_iterations:
                updates["max_iterations"] = new_max
                logger.info(
                    f"Low pass rate ({pass_rate:.2f}), increasing max_iterations to {new_max}"
                )

        # Rule 2: High conflict rate + branching disabled -> enable branching
        if conflict_rate > 0.3 and not policy.branch_trigger_on_failure:
            updates["branch_trigger_on_failure"] = True
            logger.info(
                f"High conflict rate ({conflict_rate:.2f}), enabling branch-and-merge"
            )

        # Rule 3: High pass rate + low iterations -> reduce retries to save cost
        if pass_rate > 0.8 and avg_iterations < policy.max_iterations * 0.5:
            new_max = max(policy.max_iterations - 1, 3)  # Min of 3
            if new_max < policy.max_iterations:
                updates["max_iterations"] = new_max
                logger.info(
                    f"High pass rate ({pass_rate:.2f}), reducing max_iterations to {new_max}"
                )

        # Rule 4: Branching enabled but rarely used -> adjust threshold
        if policy.branch_trigger_on_failure and branch_rate < 0.1:
            # Maybe conflicts aren't severe enough, keep current settings
            pass

        # Rule 5: Adjust spawn threshold based on conflict rate
        if conflict_rate > 0.5:
            # Many conflicts, maybe need more agents
            new_threshold = max(policy.spawn_threshold - 1, 1)
            if new_threshold < policy.spawn_threshold:
                updates["spawn_threshold"] = new_threshold
                logger.info(
                    f"High conflicts, lowering spawn_threshold to {new_threshold}"
                )

        return updates

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning progress

        Returns:
            Summary statistics
        """
        if not self.episode_history:
            return {"episodes_seen": 0}

        return {
            "episodes_seen": len(self.episode_history),
            "updates_made": self.update_count,
            "overall_pass_rate": sum(1 for e in self.episode_history if e.passed) / len(self.episode_history),
            "avg_iterations": statistics.mean(e.iterations for e in self.episode_history),
            "current_policy": self.policy_manager.get_policy_summary(),
        }

    def reset(self):
        """Reset meta-learning state"""
        self.episode_history = []
        self.update_count = 0
        logger.info("Reset meta-updater state")
