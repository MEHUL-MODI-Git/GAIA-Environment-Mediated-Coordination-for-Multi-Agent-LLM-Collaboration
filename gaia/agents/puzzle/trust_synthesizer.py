"""TrustAwareSynthesizerAgent: synthesizes a solution while down-weighting low-trust experts.

Used in the Fault Injection experiment (E9). Inherits SynthesizerAgent's parsing
+ output format but, before synthesizing, reads the trust_audit artifact posted
by the DeductionAuditorAgent. Deductions from agents with low trust scores are
labeled (or excluded) in the prompt so the LLM weighs evidence appropriately.

Behaviour:
  - If no trust_audit artifact exists, behaves like the standard SynthesizerAgent
    (graceful degradation — works in conditions without the auditor).
  - Trust < 0.4: deduction is included with a "[LOW TRUST]" warning in the prompt.
  - 0.4 ≤ trust < 0.7: deduction is included with a "[MEDIUM TRUST]" warning.
  - Trust ≥ 0.7: deduction is presented normally.

The synthesizer is told explicitly in the system prompt: "where a LOW TRUST
expert disagrees with HIGH TRUST experts, prefer the HIGH TRUST experts."
This lets us measure: does giving the synthesizer the trust signal improve
accuracy under fault injection, vs. the standard synthesizer that treats all
deductions equally?

Output: ArtifactType.REVIEW with subtype="proposed_solution" (same as standard
synthesizer so downstream Critic/Verifier are unchanged).
"""

from typing import List, Dict
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.puzzle.trust_synthesizer import TrustSynthesizerPrompts
from .synthesizer import SynthesizerAgent, parse_solution_from_text


class TrustAwareSynthesizerAgent(SynthesizerAgent):
    """Synthesizer that reads a trust score table before synthesizing.

    Args:
        people: list of person names in the puzzle.
        attributes: dict {attr_name: [possible_values]} describing the puzzle.

    The `people` and `attributes` arguments make this work on both 4×3 and 6×4
    puzzles. The trust-aware prompt format takes them explicitly so the LLM
    knows the output schema.
    """

    def __init__(self, people: list, attributes: dict,
                 partial_trust: bool = False, **kwargs):
        """
        Args:
            partial_trust: if True, use CLUE-LEVEL skepticism (keep a flagged
                expert's uncontradicted claims, discard only the specific
                conflicting ones). If False (default), use the original
                AGENT-LEVEL down-weighting. The E9 `fault_gaia_partial`
                condition sets this True to test the principled fix.
        """
        kwargs.setdefault("name", "TrustAwareSynthesizer")
        kwargs.setdefault("role", "trust_aware_synthesizer")
        super().__init__(**kwargs)
        self.people = people
        self.attributes = attributes
        self.partial_trust = partial_trust
        self.trust_prompts = TrustSynthesizerPrompts()

    def should_claim_task(self, task: Task) -> bool:
        if task.metadata.get("task_type") != "synthesize":
            return False
        root_task_id = task.parent_id or task.task_id
        existing = self.blackboard.get_artifacts_for_task(root_task_id)
        already_done = any(
            a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
            and a.author == self.agent_id
            and a.metadata.get("source_task", "") == task.task_id
            for a in existing
        )
        return not already_done

    async def execute(self, task: Task) -> List[Artifact]:
        root_task_id = task.parent_id or task.task_id

        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)

        # Find all expert deductions
        deduction_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.PLAN
            and a.metadata.get("subtype") == "partial_deduction"
        ]

        if not deduction_artifacts:
            self.logger.warning(f"{self.name}: No deductions found — cannot synthesize")
            return []

        # Find the trust audit (if any)
        audit_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.DOCUMENTATION
            and a.metadata.get("subtype") == "trust_audit"
        ]
        trust_scores: Dict[str, float] = {}
        if audit_artifacts:
            # Use the most recent audit
            latest_audit = sorted(audit_artifacts, key=lambda a: a.created_at)[-1]
            trust_scores = latest_audit.metadata.get("trust_scores", {})
            self.logger.info(
                f"{self.name}: Found trust audit with {len(trust_scores)} scored agents — "
                f"low_trust={[k[:8] for k, v in trust_scores.items() if v < 0.4]}"
            )
        else:
            self.logger.info(
                f"{self.name}: No trust audit found, treating all deductions as fully trusted"
            )

        # Build the deductions list for the prompt
        deductions_for_prompt = []
        for a in deduction_artifacts:
            deductions_for_prompt.append({
                "agent_id": a.author,
                "agent_name": a.metadata.get("agent_name", a.author[:8]),
                "content": a.content,
            })

        # Default trust=1.0 for any unscored agent
        for d in deductions_for_prompt:
            trust_scores.setdefault(d["agent_id"], 1.0)

        messages = [
            {"role": "system", "content": (
                self.trust_prompts.SYSTEM_PARTIAL if self.partial_trust
                else self.trust_prompts.SYSTEM
            )},
            {"role": "user", "content": self.trust_prompts.format_user(
                deductions=deductions_for_prompt,
                trust_scores=trust_scores,
                people=self.people,
                attributes=self.attributes,
            )},
        ]

        temperature = task.metadata.get("temperature", 0.1)
        response = await self.call_llm(messages, temperature=temperature)

        parsed = parse_solution_from_text(response)

        artifact = Artifact(
            type=ArtifactType.REVIEW,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "proposed_solution",
                "agent_name": self.name,
                "parsed_solution": parsed,
                "num_experts_read": len(deductions_for_prompt),
                "trust_scores": trust_scores,
                "used_trust_audit": bool(audit_artifacts),
                "source_task": task.task_id,
            },
        )

        self.logger.info(
            f"{self.name}: Posted trust-aware solution "
            f"(parsed={'ok' if parsed else 'FAILED'}, used_audit={bool(audit_artifacts)})"
        )
        return [artifact]
