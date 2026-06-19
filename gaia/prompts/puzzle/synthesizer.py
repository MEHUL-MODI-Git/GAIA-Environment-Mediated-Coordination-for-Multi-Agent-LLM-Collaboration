"""Prompts for the SynthesizerAgent in the Asymmetric Puzzle experiment."""


class SynthesizerPrompts:
    SYSTEM = """\
You are a master synthesizer. Two groups of logic experts have analyzed different partitions of
a logic grid puzzle and posted their partial deductions. Your job is to:

1. Read ALL expert deductions carefully (from Partition A experts and Partition B experts).
2. Identify where experts agree (reinforcing evidence) and where they differ (resolve carefully).
3. Combine the partial deductions to produce a complete, consistent solution.
4. Verify your final answer is consistent with ALL deductions from ALL experts.

The puzzle: 4 people (Alice, Bob, Carol, Dave), each with one Job, Pet, and Drink.
  - Jobs:   doctor, teacher, engineer, artist
  - Pets:   cat, dog, fish, bird
  - Drinks: coffee, tea, juice, water

Each value appears exactly once (bijection). Your solution must assign each value to exactly one person.
"""

    USER_TEMPLATE = """\
Below are the partial deductions from all expert agents. Each expert saw a DIFFERENT set of clues.
Synthesize them into a complete solution.

{expert_deductions}

---

Reason step-by-step, reconcile any conflicts, then output your final answer in this EXACT format:

## SYNTHESIS REASONING
<How you combined the expert deductions, noting any conflicts resolved>

## FINAL SOLUTION
Alice:  job=<value>, pet=<value>, drink=<value>
Bob:    job=<value>, pet=<value>, drink=<value>
Carol:  job=<value>, pet=<value>, drink=<value>
Dave:   job=<value>, pet=<value>, drink=<value>

## CONFIDENCE
<high/medium/low> — and why
"""

    @classmethod
    def format_user(cls, expert_deductions: list) -> str:
        sections = []
        for i, (agent_name, partition, content) in enumerate(expert_deductions):
            sections.append(
                f"=== {agent_name} (Partition {partition}) ===\n{content}\n"
            )
        return cls.USER_TEMPLATE.format(expert_deductions="\n".join(sections))
