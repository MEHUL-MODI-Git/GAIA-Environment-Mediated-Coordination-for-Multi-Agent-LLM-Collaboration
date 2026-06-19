"""Prompts for the PuzzleCriticAgent in the Asymmetric Puzzle experiment."""


class PuzzleCriticPrompts:
    SYSTEM = """\
You are a conflict detector. Two independent Synthesizer agents have each produced a proposed
solution to a logic grid puzzle. Your job is to:

1. Compare the two solutions attribute by attribute.
2. Identify ANY disagreements.
3. If they agree → confirm consensus.
4. If they disagree → pinpoint exactly which assignments differ and flag them.

Be precise. Do not guess which solution is correct — just identify where they differ.
"""

    USER_TEMPLATE = """\
Two synthesizers independently proposed solutions to the same logic grid puzzle.

=== SOLUTION from {synth1_name} ===
{solution1}

=== SOLUTION from {synth2_name} ===
{solution2}

---

Compare them and output in this EXACT format:

## VERDICT
<AGREE or CONFLICT>

## DIFFERENCES (if CONFLICT)
- Person <name>: {synth1_name} says <attr>=<val1>, {synth2_name} says <attr>=<val2>
(list each differing assignment; leave blank if AGREE)

## ANALYSIS
<Brief explanation of what likely caused the disagreement, if any>
"""

    @classmethod
    def format_user(
        cls,
        synth1_name: str,
        solution1: str,
        synth2_name: str,
        solution2: str,
    ) -> str:
        return cls.USER_TEMPLATE.format(
            synth1_name=synth1_name,
            solution1=solution1,
            synth2_name=synth2_name,
            solution2=solution2,
        )
