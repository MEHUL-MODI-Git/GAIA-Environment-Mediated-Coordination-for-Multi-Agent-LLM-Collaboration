"""Policy management for GAIA coordination"""

from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..blackboard.models import Policy
from ..utils.logging import get_logger

logger = get_logger("policy")


class PolicyManager:
    """Manages policy instances and updates

    Policies control GAIA coordination behavior:
    - Routing rules (which agents handle which tasks)
    - Spawning thresholds (when to create new agents)
    - Branching triggers (when to fork and try parallel solutions)
    - Verification strictness
    - Retry limits
    """

    def __init__(self, initial_policy: Optional[Policy] = None):
        self.policy = initial_policy or Policy()
        self.policy_history = [self.policy.model_copy(deep=True)]

    def get_policy(self) -> Policy:
        """Get current active policy"""
        return self.policy

    def update_policy(self, updates: Dict[str, Any]):
        """Update policy parameters

        Args:
            updates: Dictionary of policy field updates
        """
        logger.info(f"Updating policy with: {updates}")

        # Save current policy to history
        self.policy_history.append(self.policy.model_copy(deep=True))

        # Apply updates
        for key, value in updates.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                logger.info(f"Updated policy.{key} = {value}")
            else:
                logger.warning(f"Unknown policy field: {key}")

    def rollback_policy(self):
        """Rollback to previous policy version"""
        if len(self.policy_history) > 1:
            self.policy_history.pop()  # Remove current
            self.policy = self.policy_history[-1].model_copy(deep=True)
            logger.info("Rolled back policy to previous version")
        else:
            logger.warning("No previous policy to rollback to")

    def reset_policy(self):
        """Reset to default policy"""
        self.policy = Policy()
        self.policy_history = [self.policy.model_copy(deep=True)]
        logger.info("Reset policy to defaults")

    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of current policy settings"""
        return {
            "routing_rules": self.policy.routing_rules,
            "spawn_threshold": self.policy.spawn_threshold,
            "max_agents": self.policy.max_agents,
            "branch_trigger_on_failure": self.policy.branch_trigger_on_failure,
            "branch_max_parallel": self.policy.branch_max_parallel,
            "verification_strictness": self.policy.verification_strictness,
            "max_retries": self.policy.max_retries,
            "max_iterations": self.policy.max_iterations,
            "stop_on_first_pass": self.policy.stop_on_first_pass,
            "meta_update_enabled": self.policy.meta_update_enabled,
        }
