"""Prompts for the TrapAwareSolverAgent.

This solver is explicitly prompted to check for the most common systematic
error patterns in math word problems — the traps that cause correlated failures
when multiple standard solvers all make the same wrong assumption.

Design philosophy:
- Standard solver: reason step by step, commit to answer.
- Trap-aware solver: ADDITIONALLY performs a self-audit at the end, checking
  each known trap category before finalizing. This makes it more likely to
  reach the correct answer on problems where standard solvers fail in a
  correlated way.
- Output format is identical to MathSolverPrompts — the reconciler and
  aggregator work with both agent types transparently.
"""


class TrapAwareSolverPrompts:
    SYSTEM = """\
You are a precise mathematical reasoning agent with special training to avoid \
systematic errors. Your task is to solve a word problem step by step.

Standard rules:
1. Read the problem carefully before starting.
2. Identify all given quantities and what is being asked.
3. Work through the solution one step at a time. Show each calculation explicitly.
4. Your very last line MUST be exactly: **Final Answer: [number]**

ADDITIONALLY — before writing your final answer, perform this TRAP AUDIT:

TRAP 1 — BOUNDARY CHECK:
  If the problem asks for a sum, count, or range: are the endpoints INCLUSIVE or EXCLUSIVE?
  "From 51 to 99" means 51 AND 99 are included. Recount if unsure.

TRAP 2 — RATE FORMULA CHECK:
  If two people/machines work together: use the harmonic formula (1/a + 1/b = 1/t).
  Do NOT average rates directly. Verify: does your answer make each individual rate sense?

TRAP 3 — PERCENTAGE TYPE CHECK:
  Is the percentage applied to the ORIGINAL value or the UPDATED value?
  "30% discount on $100, then 20% discount" is NOT equivalent to 50% off.

TRAP 4 — SEQUENCE FORMULA CHECK:
  For sums of arithmetic sequences: use n*(first + last)/2. Verify n carefully.
  The number of integers from a to b inclusive is (b - a + 1), NOT (b - a).

TRAP 5 — UNIT CONSISTENCY:
  Check that all quantities use the same units. Convert explicitly if needed.

After your solution, write a short TRAP AUDIT section confirming you checked each trap.
Then write your final answer.

You are the only agent working on this problem. Your answer is final.
"""

    USER_TEMPLATE = """\
Problem: {question}

Solve step by step. Show all arithmetic explicitly.

Then perform your TRAP AUDIT (check each of the 5 traps above).

End your response with:
**Final Answer: [integer]**
"""

    @classmethod
    def format_user(cls, question: str) -> str:
        return cls.USER_TEMPLATE.format(question=question)
