"""Method 3: GAIA A-E (Blackboard + Verification + Conflict-as-Task)

Features:
- A: Shared blackboard
- B: Fast/slow agent tiers
- C: Self-assignment via polling
- D: Agent spawning (optional)
- E: Conflict-as-task resolution
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

logger = get_logger("method3")


class GAIAAEMethod(BaseMethod):
    """GAIA with features A through E

    Coordination via blackboard (A), tiered agents (B), self-assignment (C),
    spawning (D), and conflict-as-task resolution (E).

    Branch-and-merge (F) and meta-update (G) are disabled.
    """

    def __init__(
        self,
        coder_llm: BaseLLM,
        critic_llm: BaseLLM,
        verifier_llm: BaseLLM,
        max_iterations: int = 10,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.coder_llm = coder_llm
        self.critic_llm = critic_llm
        self.verifier_llm = verifier_llm
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.metrics = MetricsCollector()

    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve HumanEval problem using GAIA A-E

        Args:
            problem: HumanEval problem dict

        Returns:
            MethodResult with outcome and metrics
        """
        task_id = problem["task_id"]
        logger.info(f"=== Method 3: GAIA A-E for {task_id} ===")
        start_time = datetime.utcnow()

        # Create policy with Features A-E enabled, F-G disabled
        policy = Policy(
            routing_rules={
                "code_implementation": "fast",
                "code_fix": "fast",
                "review": "fast",
                "verification": "slow",
            },
            spawn_threshold=5,  # Feature D: spawn if backlog > 5
            max_agents=10,
            branch_trigger_on_failure=False,  # Feature F: DISABLED
            branch_max_parallel=0,
            verification_strictness="all_tests_pass",
            max_retries=self.max_retries,
            max_iterations=self.max_iterations,
            stop_on_first_pass=True,
            meta_update_enabled=False,  # Feature G: DISABLED
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
            f"iterations={episode_result.iterations}"
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
                "method": "gaia_ae",
                "features": "A-E",
                "artifacts_created": episode_result.artifacts_created,
                "conflicts_detected": episode_result.conflicts_detected,
                "llm_calls": llm_metrics.get("total_calls", 0),
            }
        )

    def _create_agents(self, blackboard: Blackboard) -> List[BaseAgent]:
        """Create agent team for GAIA A-E

        Args:
            blackboard: Shared blackboard

        Returns:
            List of agents
        """
        agents = [
            # Fast tier: 2 coders
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

        logger.info(f"Created {len(agents)} agents for GAIA A-E")
        return agents
