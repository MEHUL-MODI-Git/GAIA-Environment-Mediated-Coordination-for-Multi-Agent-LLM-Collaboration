"""Method 4: GAIA A-F (+ Branch-and-Merge)

Features:
- A-E: All features from Method 3
- F: Branch-and-merge sandbox trials
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
from ..episode.loop import EpisodeLoop
from ..llms.base import BaseLLM, ModelTier
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger

logger = get_logger("method4")


class GAIAAFMethod(BaseMethod):
    """GAIA with features A through F

    Adds branch-and-merge (F) to the A-E baseline:
    - When conflicts arise, fork the blackboard
    - Try N parallel solution approaches
    - Merge the best one back
    """

    def __init__(
        self,
        coder_llm: BaseLLM,
        critic_llm: BaseLLM,
        verifier_llm: BaseLLM,
        max_iterations: int = 10,
        max_retries: int = 3,
        branch_max_parallel: int = 3,
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

    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve HumanEval problem using GAIA A-F

        Args:
            problem: HumanEval problem dict

        Returns:
            MethodResult with outcome and metrics
        """
        task_id = problem["task_id"]
        logger.info(f"=== Method 4: GAIA A-F for {task_id} ===")
        start_time = datetime.utcnow()

        # Create policy with Features A-F enabled, G disabled
        policy = Policy(
            routing_rules={
                "code_implementation": "fast",
                "code_fix": "fast",
                "review": "fast",
                "verification": "slow",
            },
            spawn_threshold=5,
            max_agents=10,
            branch_trigger_on_failure=True,  # Feature F: ENABLED
            branch_max_parallel=self.branch_max_parallel,
            verification_strictness="all_tests_pass",
            max_retries=self.max_retries,
            max_iterations=self.max_iterations,
            stop_on_first_pass=True,
            meta_update_enabled=False,  # Feature G: still disabled
        )

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

        # Collect metrics
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        llm_metrics = self.metrics.get_llm_summary()

        logger.info(
            f"Episode complete: passed={episode_result.passed}, "
            f"iterations={episode_result.iterations}, "
            f"branches_created={episode_result.branches_created}"
        )

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
                "method": "gaia_af",
                "features": "A-F",
                "artifacts_created": episode_result.artifacts_created,
                "conflicts_detected": episode_result.conflicts_detected,
                "branches_created": episode_result.branches_created,
                "llm_calls": llm_metrics.get("total_calls", 0),
            }
        )

    def _create_agents(self, blackboard: Blackboard) -> List[BaseAgent]:
        """Create agent team for GAIA A-F

        Args:
            blackboard: Shared blackboard

        Returns:
            List of agents
        """
        agents = [
            # Fast tier: 2 coders (for parallel branching)
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
            # Fast tier: 1 critic
            CriticAgent(
                name="Critic-1",
                tier=ModelTier.FAST,
                llm=self.critic_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
            # Slow tier: 1 verifier
            VerifierAgent(
                name="Verifier-1",
                tier=ModelTier.SLOW,
                llm=self.verifier_llm,
                blackboard=blackboard,
                metrics=self.metrics,
            ),
        ]

        logger.info(f"Created {len(agents)} agents for GAIA A-F")
        return agents
