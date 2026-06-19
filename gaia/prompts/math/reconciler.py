"""Prompts for the MathReconcilerAgent.

Design philosophy:
- The reconciler is the most context-rich agent in the pipeline. It needs to know:
  (a) the original problem, (b) all solver reasoning chains, (c) what the conflict is.
  Armed with all three, it can pinpoint exactly where a solver went wrong.
- The reconciler IS told about the other solvers and the conflict. Unlike the solver
  (who benefits from isolation), the reconciler's entire purpose is comparative analysis.
  Knowing there was a conflict and which answers differ is essential context.
- The reconciler is asked to identify the error in incorrect solutions before giving
  its own answer. This chain-of-thought "error diagnosis" pattern significantly
  improves accuracy compared to just re-solving the problem cold.
- The reconciler uses the SAME final answer format as the solver (**Final Answer: N**)
  so the same parser can extract the answer from both UNANIMOUS and CONFLICT paths.
- The reconciler runs on the slow/more capable model (gpt-4.1 not gpt-4.1-mini)
  because it has the hardest reasoning task: auditing multiple chains for errors.
"""


class MathReconcilerPrompts:
    SYSTEM = """\
You are a mathematical reconciliation agent. You are given a word problem that was solved \
independently by multiple agents, and they disagreed on the answer. Your job is to:

1. Read the original problem carefully.
2. Read each solver's reasoning chain in full.
3. Identify exactly where each incorrect solver made an error (a specific step, \
   a misread quantity, a wrong arithmetic operation, or a misinterpretation).
4. Re-solve the problem from scratch using careful arithmetic.
5. State the correct answer.

Structure your response as:
## Error Analysis
For each solver that gave an incorrect answer, describe the exact step where they erred.

## Solution
Solve the problem step by step from scratch.

## Final Answer
Your very last line MUST be: **Final Answer: [integer]**

Be precise and authoritative. Your answer overrides all previous solver answers.
"""

    USER_TEMPLATE = """\
Problem: {question}

A conflict was detected. The solvers gave different answers:
{conflict_summary}

Here are the full reasoning chains from each solver:

{solver_outputs}

Identify the errors in the incorrect solutions, then solve the problem correctly.
End with: **Final Answer: [integer]**
"""

    @classmethod
    def format_user(
        cls,
        question: str,
        conflict_summary: str,
        solver_outputs: list,
    ) -> str:
        """
        Args:
            question: The math problem text.
            conflict_summary: E.g. "Solver-1=42, Solver-2=35, Solver-3=42"
            solver_outputs: List of (solver_name, response_text) tuples.
        """
        sections = []
        for name, response in solver_outputs:
            sections.append(f"=== {name} ===\n{response}\n")
        return cls.USER_TEMPLATE.format(
            question=question,
            conflict_summary=conflict_summary,
            solver_outputs="\n".join(sections),
        )
