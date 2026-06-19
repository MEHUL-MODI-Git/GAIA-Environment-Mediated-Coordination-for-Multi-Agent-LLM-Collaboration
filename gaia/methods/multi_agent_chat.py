"""Method 2: Multi-Agent Chat Baseline (conversational coordination)

Unlike GAIA's blackboard coordination, this method uses a debate-style
interaction where agents converse until they reach consensus or max rounds.

Pattern from AgentVerse vertical_solver_first decision maker.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseMethod, MethodResult
from ..llms.base import BaseLLM, ModelTier
from ..parsers.code_parser import CodeParser
from ..parsers.review_parser import ReviewParser
from ..execution.code_runner import CodeRunner
from ..utils.logging import get_logger

logger = get_logger("method2")


class MultiAgentChatMethod(BaseMethod):
    """Multi-agent chat baseline with solver-critic debate

    Workflow:
    1. Solver generates initial code
    2. Critic reviews code
    3. If critic agrees: test and return
    4. If critic disagrees: solver revises based on feedback
    5. Repeat until consensus or max rounds
    """

    def __init__(
        self,
        solver_llm: BaseLLM,
        critic_llm: BaseLLM,
        max_rounds: int = 5,
        max_retries: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.solver_llm = solver_llm
        self.critic_llm = critic_llm
        self.max_rounds = max_rounds
        self.max_retries = max_retries
        self.code_parser = CodeParser()
        self.review_parser = ReviewParser()
        self.code_runner = CodeRunner()

    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve HumanEval problem using multi-agent chat

        Args:
            problem: HumanEval problem dict

        Returns:
            MethodResult with outcome and metrics
        """
        task_id = problem["task_id"]
        problem_prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        logger.info(f"=== Method 2: Multi-Agent Chat for {task_id} ===")
        start_time = datetime.utcnow()

        # Track metrics
        total_tokens = 0
        total_cost = 0.0
        iterations = 0

        # Conversation history
        conversation: List[Dict[str, str]] = []
        current_code = ""
        feedback = ""

        # Try multiple times with debate rounds
        for retry in range(self.max_retries):
            logger.info(f"\n--- Retry {retry + 1} ---")

            # Debate rounds: solver <-> critic
            for round_num in range(self.max_rounds):
                logger.info(f"Round {round_num + 1}")
                iterations += 1

                # Step 1: Solver generates/revises code
                solver_result = await self._solver_turn(
                    problem_prompt, current_code, feedback, conversation
                )
                current_code = solver_result["code"]
                total_tokens += solver_result["tokens"]
                total_cost += solver_result["cost"]

                if not current_code:
                    logger.warning("Solver failed to generate code")
                    break

                # Step 2: Critic reviews code
                critic_result = await self._critic_turn(
                    problem_prompt, current_code, conversation
                )
                review = critic_result["review"]
                total_tokens += critic_result["tokens"]
                total_cost += critic_result["cost"]

                # Step 3: Check if critic agrees
                if review["agreed"]:
                    logger.info("Critic agreed! Testing solution...")

                    # Test the code
                    passed, test_output = await self.code_runner.run_humaneval_test(
                        code=current_code,
                        test=test,
                        entry_point=entry_point,
                    )

                    if passed:
                        # Success!
                        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                        logger.info(f"✓ Tests passed! (retry={retry}, round={round_num})")

                        return MethodResult(
                            task_id=task_id,
                            passed=True,
                            code=current_code,
                            iterations=iterations,
                            prompt_tokens=0,  # Aggregated in total_tokens
                            completion_tokens=0,
                            total_cost_usd=total_cost,
                            latency_ms=latency_ms,
                            metadata={
                                "method": "multi_agent_chat",
                                "rounds": round_num + 1,
                                "retries": retry + 1,
                                "consensus_reached": True,
                            }
                        )
                    else:
                        # Critic agreed but tests failed - use test output as feedback
                        feedback = f"Tests failed:\n{test_output}"
                        logger.info("Critic agreed but tests failed, retrying...")
                        break  # Move to next retry
                else:
                    # Critic disagreed - use criticism as feedback
                    feedback = review["criticism"]
                    logger.info(f"Critic disagreed: {feedback[:100]}")
                    # Continue to next round with this feedback

            # If we exhausted rounds without agreement, move to next retry
            if round_num == self.max_rounds - 1:
                logger.warning("Max rounds reached without consensus")

        # Failed to solve
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"✗ Failed to solve {task_id}")

        return MethodResult(
            task_id=task_id,
            passed=False,
            code=current_code,
            iterations=iterations,
            prompt_tokens=0,
            completion_tokens=0,
            total_cost_usd=total_cost,
            latency_ms=latency_ms,
            metadata={
                "method": "multi_agent_chat",
                "consensus_reached": False,
            }
        )

    async def _solver_turn(
        self,
        problem_prompt: str,
        current_code: str,
        feedback: str,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Solver agent's turn to generate/revise code

        Args:
            problem_prompt: HumanEval problem description
            current_code: Current code (if any)
            feedback: Feedback from critic or test failure
            conversation: Conversation history

        Returns:
            Dict with code, tokens, cost
        """
        if not current_code:
            # Initial attempt
            prompt = f"""You are a Python expert. Complete the following function:

{problem_prompt}

Provide only the complete function implementation. Use markdown code blocks."""
        else:
            # Revision based on feedback
            prompt = f"""You previously wrote this code:

```python
{current_code}
```

Feedback: {feedback}

Please revise the code to address this feedback. Provide the complete function."""

        messages = [{"role": "user", "content": prompt}]

        # Add conversation context
        for msg in conversation[-4:]:  # Last 2 exchanges
            messages.append(msg)

        # Call solver LLM
        llm_result = await self.solver_llm.agenerate(messages)

        # Parse code
        code = self.code_parser.parse(llm_result.content)

        # Add to conversation
        conversation.append({"role": "assistant", "content": llm_result.content})

        logger.info(f"Solver generated {len(code)} chars of code")

        return {
            "code": code,
            "tokens": llm_result.prompt_tokens + llm_result.completion_tokens,
            "cost": llm_result.cost_usd,
        }

    async def _critic_turn(
        self,
        problem_prompt: str,
        code: str,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Critic agent's turn to review code

        Args:
            problem_prompt: HumanEval problem description
            code: Code to review
            conversation: Conversation history

        Returns:
            Dict with review, tokens, cost
        """
        prompt = f"""You are a code reviewer. Review the following solution:

Problem:
{problem_prompt}

Solution:
```python
{code}
```

If the solution is correct and handles all edge cases, respond with [Agree].
If there are bugs or issues, explain them and end with [Disagree].

Your review:"""

        messages = [{"role": "user", "content": prompt}]

        # Add recent conversation
        for msg in conversation[-4:]:
            messages.append(msg)

        # Call critic LLM
        llm_result = await self.critic_llm.agenerate(messages)

        # Parse review
        review = self.review_parser.parse(llm_result.content)

        # Add to conversation
        conversation.append({"role": "assistant", "content": llm_result.content})

        logger.info(f"Critic {'agreed' if review['agreed'] else 'disagreed'}")

        return {
            "review": review,
            "tokens": llm_result.prompt_tokens + llm_result.completion_tokens,
            "cost": llm_result.cost_usd,
        }
