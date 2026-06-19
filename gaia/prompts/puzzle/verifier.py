"""Prompts for the PuzzleVerifierAgent (LLM sanity check pass) in the puzzle experiment."""


class PuzzleVerifierPrompts:
    SYSTEM = """\
You are a verification specialist. Given a proposed solution to a logic grid puzzle and ALL
the original clues, your job is to check whether the solution is consistent with every clue.

Check each clue methodically. Mark it PASS or FAIL. If any clue fails, the solution is WRONG.
"""

    USER_TEMPLATE = """\
Proposed solution:
{solution}

All original clues (complete list):
{all_clues}

---

Check each clue against the proposed solution:

## CLUE-BY-CLUE VERIFICATION
{clue_checks}

## VERDICT
<PASS — solution is consistent with all clues>
OR
<FAIL — solution violates clue(s): <list violated clues>>
"""

    @classmethod
    def format_user(cls, solution_text: str, all_clues: list) -> str:
        clue_lines = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(all_clues))
        checks = "\n".join(f"  Clue {i+1}: [{c}] — <check here>" for i, c in enumerate(all_clues))
        return cls.USER_TEMPLATE.format(
            solution=solution_text,
            all_clues=clue_lines,
            clue_checks=checks,
        )
