"""Method 1: Single Agent Baseline

Simple baseline: LLM generates code → run tests → retry on failure.
No multi-agent coordination, no blackboard.
"""

import time
from typing import Dict, Any
from ..llms.base import BaseLLM
from ..parsers.code_parser import CodeParser
from ..execution.code_runner import CodeRunner
from .base import BaseMethod, MethodResult


class SingleAgentMethod(BaseMethod):
    """Single-agent baseline with retry loop"""

    def __init__(self, llm: BaseLLM, max_retries: int = 3):
        """
        Args:
            llm: Language model to use
            max_retries: Maximum retry attempts
        """
        self.llm = llm
        self.max_retries = max_retries
        self.code_parser = CodeParser()
        self.code_runner = CodeRunner()

    async def solve(self, problem: Dict[str, Any]) -> MethodResult:
        """Solve problem with single-agent retry loop"""
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]

        start_time = time.time()
        total_tokens = 0
        total_cost = 0.0
        feedback = ""

        for iteration in range(1, self.max_retries + 1):
            # Build prompt
            messages = self._build_prompt(prompt, feedback, iteration)

            # Call LLM
            llm_result = await self.llm.agenerate(messages)
            total_tokens += llm_result.total_tokens
            total_cost += llm_result.cost_usd

            # Parse code
            code = self.code_parser.parse(llm_result.content)
            if not code:
                feedback = "No code found in response. Please provide a complete function implementation."
                continue

            # Run tests
            passed, test_output = await self.code_runner.run_humaneval_test(
                code, test, entry_point
            )

            # Success!
            if passed:
                latency_ms = (time.time() - start_time) * 1000
                return MethodResult(
                    task_id=task_id,
                    method="single_agent",
                    passed=True,
                    code=code,
                    iterations=iteration,
                    total_tokens=total_tokens,
                    cost_usd=total_cost,
                    latency_ms=latency_ms,
                )

            # Failed - provide feedback for next iteration
            feedback = f"Tests failed:\n{test_output}\n\nPlease fix the code."

        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000
        return MethodResult(
            task_id=task_id,
            method="single_agent",
            passed=False,
            code=code if code else "",
            iterations=self.max_retries,
            total_tokens=total_tokens,
            cost_usd=total_cost,
            latency_ms=latency_ms,
            error="Max retries exhausted",
        )

    def _build_prompt(self, problem_prompt: str, feedback: str, iteration: int) -> list:
        """Build prompt messages for LLM"""
        if iteration == 1:
            # First attempt
            return [
                {
                    "role": "user",
                    "content": f"""Complete the following Python function. Provide only the function implementation in a code block.

{problem_prompt}

Provide the complete function implementation.""",
                }
            ]
        else:
            # Retry with feedback
            return [
                {
                    "role": "user",
                    "content": f"""Complete the following Python function:

{problem_prompt}""",
                },
                {"role": "assistant", "content": "I'll provide the implementation."},
                {
                    "role": "user",
                    "content": f"""{feedback}

Please provide a corrected implementation.""",
                },
            ]
