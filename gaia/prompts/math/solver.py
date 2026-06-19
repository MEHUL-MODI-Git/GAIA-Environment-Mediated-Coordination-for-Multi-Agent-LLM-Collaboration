"""Prompts for the MathSolverAgent.

Design philosophy:
- The solver does NOT know other agents exist. This prevents anchoring — we want
  each solver to reach its answer independently through its own reasoning chain.
- The solver is told it is the sole solver and must commit to a definite answer.
  This prevents hedging ("it could be X or Y") which is unparseble downstream.
- The output format is strict: the final line MUST be "**Final Answer: [integer]**".
  Everything before that is visible to the Reconciler if a conflict occurs.
- The solver shows ALL arithmetic explicitly. This serves two purposes:
  (1) Forces deliberate step-by-step reasoning that catches errors, and
  (2) Gives the Reconciler a clear reasoning chain to audit if answers conflict.
"""


class MathSolverPrompts:
    SYSTEM = """\
You are a precise mathematical reasoning agent. Your task is to solve a word problem \
step by step and arrive at a definite integer answer.

Rules you must follow:
1. Read the problem carefully before starting.
2. Identify all given quantities and what is being asked.
3. Work through the solution one step at a time. Show each calculation explicitly \
   (e.g., "3 × 8 = 24", not just "24"). Never skip arithmetic steps.
4. Double-check your arithmetic before writing the final answer.
5. Your very last line MUST be exactly: **Final Answer: [number]**
   where [number] is a single integer (no units, no commas, no dollar signs).
6. Do NOT hedge. If you are unsure between two values, pick the one your reasoning \
   leads to and commit to it. Write one answer only.

You are the only agent working on this problem. Your answer is final.
"""

    USER_TEMPLATE = """\
Problem: {question}

Solve this step by step. Show all arithmetic explicitly. End your response with:
**Final Answer: [integer]**
"""

    @classmethod
    def format_user(cls, question: str) -> str:
        return cls.USER_TEMPLATE.format(question=question)
