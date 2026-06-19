"""Prompts for the DeductionAuditorAgent.

The Auditor reads all expert deductions posted on the blackboard and cross-checks
them for logical contradictions. It then assigns a trust score (0.0–1.0) to each
expert based on how consistent their deductions are with the rest.

Used in the Fault Injection experiment (E9). When a FaultyExpertAgent is in the
pool (one agent with 30% corrupted clues), the Auditor should flag that agent's
deductions as inconsistent and assign it a low trust score. The TrustAwareSynthesizer
then down-weights those deductions when building the final solution.
"""


class DeductionAuditorPrompts:
    SYSTEM = """\
You are a logical consistency auditor. You have been given the deductions posted \
by multiple expert agents who each analyzed a different subset of clues for a logic puzzle.

Your tasks:
1. Identify any pairs of deductions that DIRECTLY CONTRADICT each other.
   A contradiction is when Expert A says "Alice has a dog" and Expert B says "Alice has a cat".
2. For each expert, assign a TRUST SCORE from 0.0 to 1.0:
   - 1.0: all deductions are consistent with the majority, no contradictions
   - 0.5: some deductions are uncertain or hard to verify
   - 0.0: multiple deductions contradict the majority, likely has wrong information
3. Identify the SUSPECTED FAULTY EXPERT if one exists (the one with the lowest trust score
   whose deductions contradict the others).

Output your analysis in the EXACT format below. Other agents will parse it.
"""

    USER_TEMPLATE = """\
EXPERT DEDUCTIONS TO AUDIT:
{deductions_text}

---

Analyze these deductions for logical consistency, then output:

## CONTRADICTION PAIRS
(List each pair of contradicting statements — or "NONE FOUND" if no contradictions)
- Expert {agent_a}: "<statement A>" CONTRADICTS Expert {agent_b}: "<statement B>"
(repeat for each contradiction found)

## TRUST SCORES
(One line per expert, format: ExpertID: <score> — <one-sentence reason>)
{trust_score_template}

## SUSPECTED FAULTY EXPERT
(Name the expert with the lowest trust score, or "NONE" if all are consistent)
Suspected faulty: <ExpertID or NONE>
Reason: <brief explanation>

## CONSISTENCY SUMMARY
(One paragraph summarizing what you found)
"""

    @classmethod
    def format_user(cls, deductions: list) -> str:
        """
        deductions: list of dicts with keys: agent_id, agent_name, content (the deduction text)
        """
        parts = []
        trust_lines = []
        for d in deductions:
            label = d.get("agent_name", d["agent_id"][:8])
            parts.append(f"=== {label} (ID: {d['agent_id'][:8]}) ===\n{d['content']}\n")
            trust_lines.append(f"- {label}: <score> — <reason>")

        deductions_text = "\n".join(parts)
        trust_score_template = "\n".join(trust_lines)

        return cls.USER_TEMPLATE.format(
            deductions_text=deductions_text,
            trust_score_template=trust_score_template,
            agent_a="<ExpertA>",
            agent_b="<ExpertB>",
        )
