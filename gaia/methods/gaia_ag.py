"""Method 5: GAIA A-G (+ Meta-Update)

Features:
- A-F: All features from Method 4
- G: Meta-update (cross-episode policy tuning)
"""

from typing import Dict, Any, List
from datetime import datetime

from .base import BaseMethod, MethodResult
from ..blackboard.blackboard import Blackboard
from ..blackboard.storage import InMemoryStorage
from ..blackboard.models import Policy
from ..agents.base import BaseAgent
from ..agents.coder import CoderAgent
from ..agents.critic import CriticAgent
from ..agents.verifier import VerifierAgent
from ..agents.planner import PlannerAgent
from ..agents.edge_case import EdgeCaseAgent
from ..episode.loop import EpisodeLoop
from ..meta.policy import PolicyManager
from ..meta.meta_update import MetaUpdater
from ..llms.base import BaseLLM, ModelTier
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("method5")


class GAIAAGMethod(BaseMethod):
    """GAIA with all features A through G

    Adds meta-update (G) to the A-F baseline:
    - Learns from episode outcomes
    - Tunes policy parameters across episodes
    - Adapts spawn thresholds, branching triggers, retry limits
    """

    def __init__(
        self,
        coder_llm: BaseLLM,
        critic_llm: BaseLLM,
        verifier_llm: BaseLLM,
        max_iterations: int = 10,
        max_retries: int = 3,
        branch_max_parallel: int = 3,
        meta_update_frequency: int = 10,  # Update policy every N episodes
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.coder_llm = coder_llm
        self.critic_llm = critic_llm
        self.verifier_llm = verifier_llm
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.branch_max_parallel = branch_max_parallel
        self.metrics = MetricsCollector()

        # Initialize policy manager and meta-updater
        initial_policy = Policy(
            routing_rules={
                "code_implementation": "fast",
                "code_fix": "fast",
                "review": "fast",
                "verification": "slow",
            },
            spawn_threshold=5,
            max_agents=10,
            branch_trigger_on_failure=True,
            branch_max_parallel=branch_max_parallel,
            verification_strictness="all_tests_pass",
            max_retries=max_retries,
            max_iterations=max_iterations,
            stop_on_first_pass=True,
            meta_update_enabled=True,  # Feature G: ENABLED
        )

        self.policy_manager = PolicyManager(initial_policy=initial_policy)
        self.meta_updater = MetaUpdater(
            policy_manager=self.policy_manager,
            update_frequency=meta_update_frequency,
        )

    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve HumanEval problem using GAIA A-G

        Args:
            problem: HumanEval problem dict

        Returns:
            MethodResult with outcome and metrics
        """
        task_id = problem["task_id"]
        logger.info(f"=== Method 5: GAIA A-G for {task_id} ===")
        start_time = datetime.utcnow()

        # Get current policy (may have been updated by meta-learner)
        policy = self.policy_manager.get_policy()
        logger.info(f"Using policy: {self.policy_manager.get_policy_summary()}")

        # Create blackboard
        storage = InMemoryStorage()
        blackboard = Blackboard(storage=storage, policy=policy)

        # Create agents
        agents = self._create_agents(blackboard)

        # Create episode loop
        episode_loop = EpisodeLoop(
            blackboard=blackboard,
            agents=agents,
            metrics=self.metrics,
            policy=policy,
        )

        # Run episode
        episode_result = await episode_loop.run_episode(problem)

        # Record episode for meta-learning (Feature G)
        self.meta_updater.record_episode(episode_result)

        # Collect metrics
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        llm_metrics = self.metrics.get_llm_summary()

        logger.info(
            f"Episode complete: passed={episode_result.passed}, "
            f"iterations={episode_result.iterations}, "
            f"branches_created={episode_result.branches_created}"
        )

        # Get meta-learning stats
        learning_summary = self.meta_updater.get_learning_summary()

        return MethodResult(
            task_id=task_id,
            passed=episode_result.passed,
            code=episode_result.code,
            iterations=episode_result.iterations,
            prompt_tokens=llm_metrics.get("total_prompt_tokens", 0),
            completion_tokens=llm_metrics.get("total_completion_tokens", 0),
            total_cost_usd=llm_metrics.get("total_cost_usd", 0.0),
            latency_ms=latency_ms,
            metadata={
                "method": "gaia_ag",
                "features": "A-G",
                "artifacts_created": episode_result.artifacts_created,
                "conflicts_detected": episode_result.conflicts_detected,
                "branches_created": episode_result.branches_created,
                "llm_calls": llm_metrics.get("total_calls", 0),
                "meta_updates_applied": learning_summary.get("updates_made", 0),
                "overall_pass_rate": learning_summary.get("overall_pass_rate", 0.0),
            }
        )

    def _create_agents(self, blackboard: Blackboard) -> List[BaseAgent]:
        """Create enhanced agent team for GAIA A-G

        Args:
            blackboard: Shared blackboard

        Returns:
            List of agents with specialized roles
        """
        agents = [
            # Slow tier: 1 planner (for step breakdown)
            PlannerAgent(
                name="Planner-1",
                tier=ModelTier.SLOW,
                llm=self.verifier_llm,  # Use slow model
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            # Fast tier: 2 coders (parallel code generation)
            CoderAgent(
                name="Coder-1",
                tier=ModelTier.FAST,
                llm=self.coder_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            CoderAgent(
                name="Coder-2",
                tier=ModelTier.FAST,
                llm=self.coder_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            # Fast tier: 1 critic (provides feedback after verification)
            CriticAgent(
                name="Critic-1",
                tier=ModelTier.FAST,
                llm=self.critic_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            # Slow tier: 1 verifier (test execution)
            VerifierAgent(
                name="Verifier-1",
                tier=ModelTier.SLOW,
                llm=self.verifier_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            # Slow tier: 1 edge case agent (only activates on repeated failures)
            EdgeCaseAgent(
                name="EdgeCase-1",
                tier=ModelTier.SLOW,
                llm=self.verifier_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
        ]

        logger.info(f"Created {len(agents)} agents for GAIA A-G (including Planner + EdgeCase)")
        return agents

    def get_meta_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning progress

        Returns:
            Meta-learning statistics
        """
        return self.meta_updater.get_learning_summary()
