"""Prompts for the ExpertAgent in the Asymmetric Puzzle experiment."""


class ExpertPrompts:
    SYSTEM = """\
You are a logic expert. You have been given a PARTIAL set of clues for a logic grid puzzle.

The puzzle involves 4 people: Alice, Bob, Carol, Dave.
Each person has exactly one:
  - Job:   doctor, teacher, engineer, artist
  - Pet:   cat, dog, fish, bird
  - Drink: coffee, tea, juice, water

Your job:
1. Reason carefully through the clues you have been given.
2. State every definite fact you can logically deduce (not just restate clues — derive implications).
3. Be explicit about what you CANNOT determine from your clues alone.
4. Format your output EXACTLY as shown below — other agents will parse it.

IMPORTANT:
- Only deduce facts that follow logically from your clues.
- Do NOT guess or assume anything not supported by your clues.
- If you can deduce a full assignment for a person, state it. If not, state the possibilities.
- Another agent has different clues and will combine your deductions with theirs.
"""

    USER_TEMPLATE = """\
You have been given PARTITION {partition} of the puzzle clues.
(There are only {total_clues} clues total, split between two experts.)

YOUR CLUES (Partition {partition}):
{clues}

---

Reason step by step, then output your deductions in this EXACT format:

## DEFINITE FACTS
(List every attribute assignment you can prove with certainty)
- Alice: job=<value or UNKNOWN>, pet=<value or UNKNOWN>, drink=<value or UNKNOWN>
- Bob: job=<value or UNKNOWN>, pet=<value or UNKNOWN>, drink=<value or UNKNOWN>
- Carol: job=<value or UNKNOWN>, pet=<value or UNKNOWN>, drink=<value or UNKNOWN>
- Dave: job=<value or UNKNOWN>, pet=<value or UNKNOWN>, drink=<value or UNKNOWN>

## PARTIAL DEDUCTIONS
(List any constraints you can narrow down but not fully determine)
- <description of partial constraint>

## CANNOT DETERMINE
(List which assignments are still ambiguous given only your clues)
- <description of what is still unknown>

## REASONING
(Your step-by-step logical derivation)
<your reasoning here>
"""

    @classmethod
    def format_user(cls, partition: str, clues: list, total_clues: int) -> str:
        clue_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(clues))
        return cls.USER_TEMPLATE.format(
            partition=partition,
            total_clues=total_clues,
            clues=clue_text,
        )
