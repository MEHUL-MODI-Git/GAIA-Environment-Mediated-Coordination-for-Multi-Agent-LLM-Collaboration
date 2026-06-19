"""Prompts for the TrustAwareSynthesizerAgent.

Like the standard SynthesizerAgent but receives a trust score table alongside
the expert deductions. Deductions from low-trust experts are labeled with
warnings. The synthesizer uses this to weight evidence appropriately.

Used in the Fault Injection experiment (E9) alongside the DeductionAuditorAgent.
"""


class TrustSynthesizerPrompts:
    SYSTEM = """\
You are a logic puzzle synthesizer. You have received deductions from multiple \
expert agents. Some experts have been assigned TRUST SCORES by an auditor — \
lower scores indicate the expert may have received incorrect or corrupted information.

Your tasks:
1. Read ALL expert deductions carefully.
2. Pay special attention to TRUST SCORES. Deductions from LOW TRUST experts \
   (score < 0.5) may be based on wrong information — treat them as hints, not facts.
3. Where experts AGREE, treat that as strong evidence.
4. Where a LOW TRUST expert DISAGREES with HIGH TRUST experts, prefer the HIGH TRUST experts.
5. Produce the complete, definite solution.

Output format is EXACT — other agents parse it. Show your reasoning before the solution.
"""

    # Principled variant: CLUE-LEVEL skepticism instead of AGENT-LEVEL exclusion.
    # A low-trust expert may have received only PARTIALLY corrupted information —
    # most of its deductions are still correct and, in an information-asymmetric
    # puzzle, may be the ONLY source of some necessary clues. Excluding the whole
    # expert destroys that necessary information. This prompt keeps every claim
    # that is corroborated or uncontradicted, and discards ONLY the specific
    # claims that directly conflict with higher-trust experts.
    SYSTEM_PARTIAL = """\
You are a logic puzzle synthesizer. You have received deductions from multiple \
expert agents. Some experts have been assigned TRUST SCORES by an auditor — a \
lower score means SOME of that expert's input may be corrupted, but typically \
MOST of its deductions are still correct and may be the only source of certain \
necessary facts.

Your tasks:
1. Read ALL expert deductions carefully — including low-trust ones.
2. Treat trust scores as CLAIM-LEVEL skepticism, NOT agent-level exclusion:
   - Keep every claim from a low-trust expert that is corroborated by, or not \
     contradicted by, the other experts.
   - Discard ONLY the specific individual claims from a low-trust expert that \
     DIRECTLY CONTRADICT higher-trust experts.
   - Never discard an entire expert's contribution — you would lose necessary \
     information that no other expert has.
3. Where experts AGREE, treat that as strong evidence.
4. Reconstruct the complete solution using all surviving claims.
5. Produce the complete, definite solution.

Output format is EXACT — other agents parse it. Show your reasoning before the solution.
"""

    USER_TEMPLATE = """\
EXPERT DEDUCTIONS WITH TRUST SCORES:
{deductions_with_trust}

---

PUZZLE ATTRIBUTES:
People: {people}
Attributes: {attributes}

Synthesize a complete solution. Show your reasoning. Output:

## SOLUTION
{solution_template}

## REASONING
<explain how you resolved any contradictions and which trust scores influenced your decision>
"""

    @classmethod
    def format_user(
        cls,
        deductions: list,
        trust_scores: dict,
        people: list,
        attributes: dict,
    ) -> str:
        """
        deductions: list of dicts {agent_id, agent_name, content}
        trust_scores: dict {agent_id: float}
        people: list of person names
        attributes: dict {attr_name: [possible_values]}
        """
        parts = []
        for d in deductions:
            label = d.get("agent_name", d["agent_id"][:8])
            score = trust_scores.get(d["agent_id"], 1.0)
            trust_tag = ""
            if score < 0.4:
                trust_tag = " ⚠️ [LOW TRUST — treat with skepticism]"
            elif score < 0.7:
                trust_tag = " [MEDIUM TRUST — verify against others]"
            else:
                trust_tag = " [HIGH TRUST]"
            parts.append(
                f"=== {label} (Trust: {score:.1f}{trust_tag}) ===\n{d['content']}\n"
            )

        sol_lines = []
        for person in people:
            attr_parts = ", ".join(f"{attr}=<value>" for attr in attributes)
            sol_lines.append(f"- {person}: {attr_parts}")

        return cls.USER_TEMPLATE.format(
            deductions_with_trust="\n".join(parts),
            people=", ".join(people),
            attributes=", ".join(attributes.keys()),
            solution_template="\n".join(sol_lines),
        )
