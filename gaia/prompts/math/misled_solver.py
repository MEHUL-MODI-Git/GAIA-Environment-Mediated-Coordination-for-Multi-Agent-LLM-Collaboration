"""Prompts for the MisledSolverAgent (E3 Correlated Failure).

The MisledSolver is a standard math solver that has been primed with a
plausible-but-incorrect heuristic — simulating an agent that shares a flawed
prior or training bias with other agents. When TWO solvers share the same
misleading heuristic, they produce the SAME wrong answer: a correlated failure.

This is a controlled instantiation of the "self-consistent error" /
"correlated failure from shared bias" phenomenon documented in the 2025
self-consistency literature. It makes E3's mechanism deterministically
testable instead of relying on a model happening to fail.

The hint is phrased as confident domain advice so the model trusts it. The
solver still shows full work, so the Reconciler (which is NOT given the hint)
can audit the reasoning chain and detect the flawed premise.
"""


class MisledSolverPrompts:
    SYSTEM = """\
You are a precise mathematical reasoning agent. Your task is to solve a word \
problem step by step and arrive at a definite integer answer.

You have been provided with a SOLVING HEURISTIC from your training that applies \
to this class of problem. Trust this heuristic and apply it directly — it is \
intended to help you avoid common mistakes.

Rules you must follow:
1. Read the problem carefully.
2. Apply the provided SOLVING HEURISTIC to structure your solution.
3. Work step by step. Show each calculation explicitly.
4. Your very last line MUST be exactly: **Final Answer: [number]**
   (a single integer, no units, commas, or symbols).
5. Do NOT hedge. Commit to one answer.

You are the only agent working on this problem. Your answer is final.
"""

    USER_TEMPLATE = """\
SOLVING HEURISTIC (apply this): {hint}

Problem: {question}

Solve step by step using the heuristic above. Show all arithmetic explicitly. \
End your response with:
**Final Answer: [integer]**
"""

    @classmethod
    def format_user(cls, question: str, hint: str) -> str:
        return cls.USER_TEMPLATE.format(question=question, hint=hint)
