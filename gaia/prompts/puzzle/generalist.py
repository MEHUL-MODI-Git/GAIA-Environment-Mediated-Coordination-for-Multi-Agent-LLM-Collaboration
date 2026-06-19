"""Prompts for the GeneralistAgent.

The GeneralistAgent is used as the homogeneous baseline in the Agent Scaling
experiment (E8). Unlike ExpertAgent (which sees a partition), the Generalist
sees ALL clues and attempts a complete solution independently. Multiple
Generalists run in parallel and majority-vote their answers — with no
blackboard coordination between them.

This tests: does naive horizontal scaling (N identical agents voting) perform
as well as GAIA's role-specialized coordination? The hypothesis: it does not,
because adding identical agents doesn't add new information.
"""


class GeneralistPrompts:
    SYSTEM = """\
You are a logic puzzle solver. You will be given a complete set of clues for a \
logic grid puzzle. Your task is to deduce the full solution.

The puzzle involves {n_people} people. Each person has exactly one value for each attribute.

Your job:
1. Read all clues carefully.
2. Use deductive reasoning to eliminate impossible assignments.
3. Derive the complete solution — every person's full assignment.
4. Output your solution in the EXACT format specified below.

Rules:
- Only state facts you can prove with certainty from the clues.
- Do NOT guess. If you cannot determine an attribute, say UNKNOWN.
- Show your reasoning step by step before the final answer.
"""

    USER_TEMPLATE = """\
PUZZLE CLUES ({n_clues} total):
{clues}

---

Reason step by step through the clues. Then output your solution in this EXACT format:

## SOLUTION
{solution_template}

## REASONING
<your step-by-step deduction here>
"""

    @classmethod
    def format_user(cls, clues: list, people: list, attributes: dict) -> str:
        clue_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(clues))
        sol_lines = []
        for person in people:
            attr_parts = ", ".join(
                f"{attr}=<value>" for attr in attributes
            )
            sol_lines.append(f"- {person}: {attr_parts}")
        solution_template = "\n".join(sol_lines)
        return cls.USER_TEMPLATE.format(
            n_clues=len(clues),
            clues=clue_text,
            solution_template=solution_template,
        )

    @classmethod
    def format_system(cls, n_people: int) -> str:
        return cls.SYSTEM.format(n_people=n_people)
