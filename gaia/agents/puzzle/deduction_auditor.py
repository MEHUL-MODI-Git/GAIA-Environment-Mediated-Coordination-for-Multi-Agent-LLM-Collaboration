"""DeductionAuditorAgent: cross-checks expert deductions for logical contradictions.

Used in the Fault Injection experiment (E9). After all ExpertAgents (including
the FaultyExpertAgent) post their partial deductions, this agent claims a
"deduction_audit" task. It reads all deductions from the blackboard, asks an
LLM to identify contradicting pairs and assign per-agent trust scores, then:

  1. Posts a DOCUMENTATION artifact containing the full trust score table.
     The TrustAwareSynthesizerAgent reads this artifact to weight inputs.

  2. If contradictions are found, posts a CONFLICT signal so the episode loop
     can react (e.g., spawn re-deduction tasks or branch the synthesis).

The key insight: a faulty expert's deductions will logically contradict the
correct experts' deductions. By cross-checking statements at the deduction
level (before synthesis), we can identify which expert is unreliable WITHOUT
knowing the ground truth.

Output:
  - ArtifactType.DOCUMENTATION with subtype="trust_audit" + metadata containing
    parsed trust scores and contradiction pairs.
  - Optional Signal(type=CONFLICT) if contradictions are found.
"""

import re
from typing import List, Dict, Optional, Tuple
from ...blackboard.models import (
    Task, Artifact, ArtifactType, Signal, SignalType,
)
from ...prompts.puzzle.deduction_auditor import DeductionAuditorPrompts
from ..base import BaseAgent


def parse_trust_scores(text: str, agent_id_lookup: Dict[str, str]) -> Dict[str, float]:
    """Extract per-agent trust scores from the auditor's response.

    The auditor outputs lines like:
        - Expert-A: 0.9 — consistent with majority
        - FaultyExpert-B: 0.2 — contradicts other deductions on Carol's pet

    Args:
        text: full auditor response.
        agent_id_lookup: dict {agent_name: agent_id} so we can map names back
            to canonical agent IDs (the trust dict is keyed by agent_id for
            the synthesizer).

    Returns:
        dict {agent_id: score}. Missing agents default to 1.0 (full trust).
    """
    scores: Dict[str, float] = {}
    # Match lines like "- Name: 0.9 — reason" or "Name: 0.9 because ..."
    pattern = re.compile(
        r"[-*\s]*([A-Za-z][\w\-\.]*)\s*[:\-]\s*"
        r"(?:trust\s*[:=])?\s*(\d*\.?\d+)",
        re.IGNORECASE,
    )

    # Only look at the TRUST SCORES section to avoid false matches
    section_match = re.search(
        r"##?\s*TRUST\s*SCORES\s*(.*?)(?:##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    section = section_match.group(1) if section_match else text

    for line in section.splitlines():
        m = pattern.search(line.strip())
        if not m:
            continue
        name = m.group(1)
        try:
            score = float(m.group(2))
        except ValueError:
            continue
        if not (0.0 <= score <= 1.0):
            continue
        agent_id = agent_id_lookup.get(name)
        if agent_id:
            scores[agent_id] = score

    return scores


def parse_contradiction_pairs(text: str) -> List[str]:
    """Extract the list of contradicting deduction pairs from the response."""
    section_match = re.search(
        r"##?\s*CONTRADICTION\s*PAIRS\s*(.*?)(?:##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not section_match:
        return []
    section = section_match.group(1).strip()
    if "NONE FOUND" in section.upper():
        return []
    pairs = []
    for line in section.splitlines():
        line = line.strip().lstrip("-*").strip()
        if line and "CONTRADICTS" in line.upper():
            pairs.append(line)
    return pairs


def parse_suspected_faulty(text: str) -> Optional[str]:
    """Extract the suspected faulty agent name from the response."""
    m = re.search(
        r"Suspected\s*faulty\s*:\s*([A-Za-z][\w\-\.]*)",
        text,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip()
        if name.upper() == "NONE":
            return None
        return name
    return None


class DeductionAuditorAgent(BaseAgent):
    """Audits expert deductions and posts a trust score table.

    Reads all partial_deduction artifacts → asks LLM to cross-check →
    posts trust scores as a DOCUMENTATION artifact.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "DeductionAuditor")
        kwargs.setdefault("role", "deduction_auditor")
        super().__init__(**kwargs)
        self.prompts = DeductionAuditorPrompts()

    def should_claim_task(self, task: Task) -> bool:
        if task.metadata.get("task_type") != "deduction_audit":
            return False
        # Don't re-execute if we already posted an audit for this puzzle
        root_task_id = task.parent_id or task.task_id
        existing = self.blackboard.get_artifacts_for_task(root_task_id)
        already_done = any(
            a.type == ArtifactType.DOCUMENTATION
            and a.metadata.get("subtype") == "trust_audit"
            and a.author == self.agent_id
            for a in existing
        )
        return not already_done

    async def execute(self, task: Task) -> List[Artifact]:
        root_task_id = task.parent_id or task.task_id

        # Read all partial deductions from blackboard
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        deduction_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
        ]

        if len(deduction_artifacts) < 2:
            self.logger.warning(
                f"{self.name}: Only {len(deduction_artifacts)} deduction(s) — "
                "cannot audit cross-consistency"
            )
            return []

        # Build the input list and name→id lookup
        deductions = []
        agent_id_lookup: Dict[str, str] = {}
        for a in deduction_artifacts:
            agent_name = a.metadata.get("agent_name", a.author[:8])
            deductions.append({
                "agent_id": a.author,
                "agent_name": agent_name,
                "content": a.content,
            })
            agent_id_lookup[agent_name] = a.author

        self.logger.info(
            f"{self.name}: Auditing {len(deductions)} expert deductions for consistency"
        )

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(deductions)},
        ]

        response = await self.call_llm(messages, temperature=0.0)

        # Parse the structured output
        trust_scores = parse_trust_scores(response, agent_id_lookup)
        contradictions = parse_contradiction_pairs(response)
        suspected_name = parse_suspected_faulty(response)
        suspected_id = agent_id_lookup.get(suspected_name) if suspected_name else None

        # Fill in default trust=1.0 for any expert the LLM didn't score
        for a in deduction_artifacts:
            trust_scores.setdefault(a.author, 1.0)

        self.logger.info(
            f"{self.name}: trust_scores={ {n: round(trust_scores.get(i, 1.0), 2) for n, i in agent_id_lookup.items()} } "
            f"contradictions={len(contradictions)} suspected={suspected_name}"
        )

        # Post trust audit as DOCUMENTATION artifact for the synthesizer to read
        audit_artifact = Artifact(
            type=ArtifactType.DOCUMENTATION,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "trust_audit",
                "agent_name": self.name,
                "trust_scores": trust_scores,
                "agent_id_lookup": agent_id_lookup,
                "contradictions": contradictions,
                "suspected_faulty_agent_id": suspected_id,
                "suspected_faulty_agent_name": suspected_name,
                "n_audited": len(deductions),
            },
        )

        # If we found contradictions, raise a CONFLICT signal so the episode
        # loop knows something is off (useful for logging + visualization).
        if contradictions:
            signal = Signal(
                type=SignalType.CONFLICT,
                task_id=root_task_id,
                description=(
                    f"Auditor found {len(contradictions)} contradicting deduction pair(s). "
                    f"Suspected faulty: {suspected_name or 'unclear'}"
                ),
                severity=0.9,
                metadata={
                    "contradictions": contradictions[:5],  # cap for log size
                    "suspected_faulty_agent_id": suspected_id,
                    "trust_scores": trust_scores,
                    "source": "deduction_auditor",
                },
            )
            self.blackboard.post_signal(signal)
            self.logger.warning(
                f"{self.name}: Posted CONFLICT signal — {len(contradictions)} contradictions found"
            )

        return [audit_artifact]
