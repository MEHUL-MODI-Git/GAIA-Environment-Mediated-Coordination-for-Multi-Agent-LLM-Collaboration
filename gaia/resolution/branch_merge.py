"""Branch-and-merge mechanism (Feature F: Sandbox Trials)"""

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import Task, Artifact, ArtifactType, Evidence
from ..agents.base import BaseAgent
from ..utils.blackboard_logger import EventType
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger
from ..execution.code_runner import CodeRunner

logger = get_logger("branch_merge")


_DIVERSITY_HINTS = [
    "Use a completely different algorithm than the obvious approach. Think about the problem from scratch.",
    "Focus especially on boundary and edge cases: empty inputs, single elements, negative numbers, zero.",
    "Prioritize correctness over cleverness. Use simple loops and explicit conditions rather than one-liners.",
]


class BranchManager:
    """Manages parallel solution attempts via blackboard forking

    Feature F: When a task fails or gets stuck, fork the blackboard,
    try multiple approaches in parallel, and merge the best solution.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        agents: List[BaseAgent],
        metrics: MetricsCollector,
        budget_monitor=None,
    ):
        self.blackboard = blackboard
        self.agents = agents
        self.metrics = metrics
        self.budget_monitor = budget_monitor
        self.code_runner = CodeRunner()

    async def branch_and_merge(
        self,
        root_task: Task,
        problem: Dict[str, Any],
        n_branches: int = 3,
        planner=None,
        failure_summary: str = "",
    ) -> Optional[str]:
        """Fork blackboard, run parallel solution attempts, merge winner

        Algorithm:
        1. (Optional) Ask Planner to generate N problem-specific approaches
        2. Create N forked blackboards
        3. Run agents independently on each fork with approach-specific diversity hints
        4. Evaluate each solution
        5. Merge the best solution back to main blackboard

        Args:
            root_task: Task to solve
            problem: HumanEval problem dict with test/entry_point
            n_branches: Number of parallel attempts
            planner: Optional PlannerAgent — if provided, generates problem-specific approaches
            failure_summary: Latest test failure output (passed to planner for context)

        Returns:
            Winning code if any solution passes, None otherwise
        """
        logger.info(f"Starting branch-and-merge with {n_branches} parallel attempts")

        # Step 1: Generate diversity hints — problem-specific if planner available
        if planner is not None and failure_summary:
            try:
                diversity_hints = await planner.generate_approaches(
                    problem_prompt=root_task.description,
                    failure_summary=failure_summary,
                    n=n_branches,
                )
                logger.info("BranchManager: Using Planner-generated approaches as diversity hints")
            except Exception as exc:
                logger.warning(f"BranchManager: Planner approach generation failed ({exc}), using defaults")
                diversity_hints = _DIVERSITY_HINTS[:n_branches]
        else:
            diversity_hints = _DIVERSITY_HINTS[:n_branches]

        # Step 2: Create forked blackboards
        forks = []
        for i in range(n_branches):
            fork_id = f"branch_{root_task.task_id}_{i}"
            fork = self.blackboard.fork(fork_id)
            forks.append((fork_id, fork))
            logger.info(f"Created fork: {fork_id}")
            self.blackboard.logger.log_event(
                EventType.BRANCH_CREATED,
                actor="branch_merge",
                details={
                    "fork_id": fork_id,
                    "branch_index": i,
                    "n_branches": n_branches,
                    "diversity_hint": diversity_hints[i % len(diversity_hints)],
                    "root_task_id": root_task.task_id,
                },
            )

        # Step 3: Run agents on each fork in parallel, with diversity hints
        branch_tasks = []
        for i, (fork_id, fork_bb) in enumerate(forks):
            diversity_hint = diversity_hints[i % len(diversity_hints)]
            task = self._run_branch(fork_id, fork_bb, root_task, problem, diversity_hint)
            branch_tasks.append(task)

        branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)

        # Step 4: Evaluate solutions
        winners = []
        for i, result in enumerate(branch_results):
            fork_id, fork_bb = forks[i]

            if isinstance(result, Exception):
                logger.error(f"Branch {fork_id} failed with error: {result}")
                continue

            code, passed, test_output = result

            if passed:
                winners.append({
                    "fork_id": fork_id,
                    "fork_bb": fork_bb,
                    "code": code,
                    "test_output": test_output,
                })
                logger.info(f"Branch {fork_id} produced passing solution!")

        # Step 5: Merge best solution
        if not winners:
            logger.warning("No branches produced passing solutions")
            self.metrics.record_merge(success=False)
            all_fork_results = []
            for i, result in enumerate(branch_results):
                fork_id = forks[i][0]
                if isinstance(result, Exception):
                    all_fork_results.append({"fork_id": fork_id, "passed": False, "error": str(result)})
                else:
                    code, passed, test_output = result
                    all_fork_results.append({"fork_id": fork_id, "passed": passed,
                                             "test_summary": test_output[:200] if test_output else ""})
            self.blackboard.logger.log_event(
                EventType.BRANCH_MERGED,
                actor="branch_merge",
                details={
                    "winning_fork": None,
                    "all_results": all_fork_results,
                    "n_branches": n_branches,
                    "passed": False,
                    "root_task_id": root_task.task_id,
                },
            )
            return None

        # Take first winner (could add ranking logic here)
        winner = winners[0]
        logger.info(f"Merging winner from {winner['fork_id']}")

        # Merge the winning fork back to main blackboard
        self.blackboard.merge(winner["fork_bb"])

        self.metrics.record_branch(n_branches=n_branches)
        self.metrics.record_merge(success=True)

        all_fork_results = []
        for i, result in enumerate(branch_results):
            fork_id = forks[i][0]
            if isinstance(result, Exception):
                all_fork_results.append({"fork_id": fork_id, "passed": False, "error": str(result)})
            else:
                code, passed, test_output = result
                all_fork_results.append({"fork_id": fork_id, "passed": passed,
                                         "test_summary": test_output[:200] if test_output else ""})

        self.blackboard.logger.log_event(
            EventType.BRANCH_MERGED,
            actor="branch_merge",
            details={
                "winning_fork": winner["fork_id"],
                "all_results": all_fork_results,
                "n_branches": n_branches,
                "passed": True,
                "root_task_id": root_task.task_id,
            },
        )

        return winner["code"]

    async def _run_branch(
        self,
        fork_id: str,
        fork_bb: Blackboard,
        root_task: Task,
        problem: Dict[str, Any],
        diversity_hint: str = "",
    ) -> tuple[str, bool, str]:
        """Run agents on a forked blackboard with a diversity hint.

        Args:
            fork_id: Identifier for this branch
            fork_bb: Forked blackboard instance
            root_task: Task to solve
            problem: HumanEval problem
            diversity_hint: Instruction to steer this branch toward a different approach

        Returns:
            Tuple of (code, passed, test_output)
        """
        logger.info(f"Running branch {fork_id} | hint: {diversity_hint[:60]}")

        fork_agents = self._create_fork_agents(fork_bb)

        # Inject diversity hint into the task metadata so Coder reads it
        fork_task = Task(
            title=root_task.title,
            description=root_task.description,
            acceptance_criteria=root_task.acceptance_criteria,
            metadata={
                **root_task.metadata,
                "branch_id": fork_id,
                "branch_attempt": True,
                "diversity_hint": diversity_hint,
                # Pass existing feedback from main board so branches benefit from it
                "feedback": root_task.metadata.get("feedback", ""),
            },
        )
        fork_bb.post_task(fork_task)

        # More iterations per branch than before (3 → 5)
        max_iterations = 5

        for iteration in range(max_iterations):
            # Run all agents one iteration
            await asyncio.gather(
                *[agent.run_loop(max_iterations=1) for agent in fork_agents]
            )

            # Check if we have a solution
            latest_code = fork_bb.get_latest_artifact(
                fork_task.task_id, ArtifactType.CODE
            )

            if latest_code:
                # Test it
                passed, test_output = await self.code_runner.run_humaneval_test(
                    code=latest_code.content,
                    test=problem["test"],
                    entry_point=problem["entry_point"],
                )

                if passed:
                    logger.info(f"Branch {fork_id} found solution in iteration {iteration}")
                    return latest_code.content, True, test_output

        # No passing solution found
        latest_code = fork_bb.get_latest_artifact(fork_task.task_id, ArtifactType.CODE)
        if latest_code:
            # Return best attempt even if it fails
            _, test_output = await self.code_runner.run_humaneval_test(
                code=latest_code.content,
                test=problem["test"],
                entry_point=problem["entry_point"],
            )
            return latest_code.content, False, test_output

        return "", False, "No code generated"

    def _create_fork_agents(self, fork_bb: Blackboard) -> List[BaseAgent]:
        """Create agent instances for a forked blackboard.

        Passes budget_monitor if available so cost tracking works across branches.
        """
        from ..agents.planner import PlannerAgent
        fork_agents = []

        for agent in self.agents:
            # Skip Planner in branches — no need for re-planning
            if isinstance(agent, PlannerAgent):
                continue
            fork_agent = agent.__class__(
                name=agent.name,
                role=agent.role,
                tier=agent.tier,
                llm=agent.llm,           # Share LLM (stateless)
                blackboard=fork_bb,       # Point to fork
                metrics=self.metrics,     # Share metrics
                budget_monitor=self.budget_monitor,  # Share budget tracking
            )
            fork_agents.append(fork_agent)

        return fork_agents
