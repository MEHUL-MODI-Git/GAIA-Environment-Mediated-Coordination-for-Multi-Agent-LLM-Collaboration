"""Prompts for the MathAggregatorAgent.

Design philosophy:
- The aggregator knows it received solutions from N independent solvers.
  It does NOT re-solve the problem. Its only job is consensus detection.
- Knowing the solver count and identities lets it report conflicts precisely
  ("Solver-1=42 vs Solver-2=35 vs Solver-3=42") so the Reconciler knows exactly
  which solvers to trust and which to audit.
- The output format is strictly either UNANIMOUS or CONFLICT so the episode loop
  can parse it with a simple regex and post the right signal.
- The aggregator is explicitly told NOT to evaluate reasoning quality — it only
  looks at final answers. This prevents it from trying to do the Reconciler's job.
"""


class MathAggregatorPrompts:
    SYSTEM = """\
You are a consensus-detection agent. Multiple independent solvers have each worked on \
the same math problem and submitted their answers. Your job is to check whether they agree.

Rules:
1. Extract the final integer answer from each solver's submission. The final answer \
   is always on a line that starts with "**Final Answer:".
2. Compare the extracted integers. Do NOT re-solve the problem yourself.
3. If all solvers give the same integer, output: UNANIMOUS: [integer]
4. If any solvers disagree, output: CONFLICT: [list each solver and their answer]
   Format: CONFLICT: Solver-1=[X], Solver-2=[Y], Solver-3=[Z]

Output ONLY one of these two formats. No explanation needed.
"""

    USER_TEMPLATE = """\
Problem: {question}

The following independent solvers have each submitted their solution:

{solver_outputs}

Do the solvers agree on the final answer?
Output UNANIMOUS: [integer] if they all agree, or CONFLICT: Solver-1=[X], ... if they disagree.
"""

    @classmethod
    def format_user(cls, question: str, solver_outputs: list) -> str:
        """
        Args:
            question: The math problem text.
            solver_outputs: List of (solver_name, response_text) tuples.
        """
        sections = []
        for name, response in solver_outputs:
            sections.append(f"--- {name} ---\n{response}\n")
        return cls.USER_TEMPLATE.format(
            question=question,
            solver_outputs="\n".join(sections),
        )
