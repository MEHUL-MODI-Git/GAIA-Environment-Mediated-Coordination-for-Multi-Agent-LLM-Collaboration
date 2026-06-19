"""Coder agent - generates code implementations"""

from typing import List
from ..blackboard.models import Task, Artifact, ArtifactType
from ..prompts.coder import CoderPrompts
from ..parsers.code_parser import CodeParser
from .base import BaseAgent


class CoderAgent(BaseAgent):
    """Fast-tier agent that generates code implementations.

    Bug fixes vs previous version:
    1. Posts artifacts under ROOT task id (task.parent_id or task.task_id)
       so the episode loop can always find the latest code.
    2. Reads Critic claims from root task scope (not fix_task scope where they don't exist).
    3. Falls back to task.metadata["feedback"] and task.metadata["previous_code"]
       when blackboard data is unavailable (for fix tasks).
    4. Includes micro-checker warnings in prompt when available.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "Coder")
        kwargs.setdefault("role", "coder")
        super().__init__(**kwargs)
        self.prompts = CoderPrompts()
        self.parser = CodeParser()

    def should_claim_task(self, task: Task) -> bool:
        """Coder should claim tasks that don't have code yet.

        Uses task.task_id scope (not root_task_id) so that:
        - Root task: skipped if code exists (Verifier should run)
        - Fix tasks: always claimable (they have no code under their own task_id)
        Code is then posted under root_task_id in execute().

        When FIX_DIRECTIVES exist from the Critic, the Fixer agent handles
        targeted repair instead — Coder defers to avoid re-anchoring on wrong code.
        """
        code_artifact = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.CODE)
        if code_artifact:
            return False
        # If Critic has posted FIX_DIRECTIVES, let Fixer handle it
        root_task_id = task.parent_id or task.task_id
        claims = self.blackboard.storage.get_claims_for_task(root_task_id)
        for c in reversed(claims):
            if "fix_directives" in c.statement.lower():
                self.logger.info(f"{self.name}: Deferring to Fixer (FIX_DIRECTIVES present)")
                return False
        return True

    async def execute(self, task: Task) -> List[Artifact]:
        """Generate or refine code for the task.

        Args:
            task: Task containing problem description. May be root task or fix task.

        Returns:
            List containing CODE artifact posted under root task id.
        """
        problem_prompt = task.description

        # ===== DETERMINE ROOT TASK SCOPE =====
        # Fix tasks have parent_id pointing to the root task.
        # We always store code under root task id so the episode loop can find it.
        root_task_id = task.parent_id or task.task_id

        # ===== READ PRIOR CODE (root scope) =====
        prior_code_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.CODE)
        prior_code = prior_code_artifact.content if prior_code_artifact else ""

        # Fallback: read from task.metadata (set by episode loop's _resolve_conflicts)
        if not prior_code:
            prior_code = task.metadata.get("previous_code", "")

        # ===== READ CRITIC FEEDBACK (root scope) =====
        # Critic posts claims under root_task_id. Take the latest failure claim only —
        # the most recent critic analysis is the most actionable.
        all_claims = self.blackboard.storage.get_claims_for_task(root_task_id)
        failure_claims = [
            c for c in all_claims
            if any(kw in c.statement.lower()
                   for kw in ["fix_directives", "failure_hypothesis"])
        ]
        # Most recent claim first — avoid drowning in old contradictory feedback
        feedback = failure_claims[-1].statement if failure_claims else ""

        # Fallback: read from task.metadata["feedback"] (set by episode loop)
        if not feedback:
            feedback = task.metadata.get("feedback", "")

        # ===== READ PLANNER HINTS (only on first attempt, before any code exists) =====
        plan_hint = ""
        if not prior_code:
            plan_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.PLAN)
            if plan_artifact:
                plan_hint = plan_artifact.content

        # ===== LOOP DETECTION: track attempted code hashes =====
        attempted_hashes: set = set()
        all_code_arts = self.blackboard.get_artifacts_for_task(root_task_id)
        for a in all_code_arts:
            if a.type == ArtifactType.CODE:
                attempted_hashes.add(hash(a.content.strip()))

        # ===== READ MICRO-CHECKER WARNINGS =====
        micro_warnings = task.metadata.get("micro_checker_warnings", "")

        # ===== READ TEST FAILURE EVIDENCE (from root task's code) =====
        failure_info = ""
        if prior_code_artifact:
            fail_evidence = [
                e for e in self.blackboard.get_evidence_for_artifact(prior_code_artifact.artifact_id)
                if e.passed is False
            ]
            failure_info = "\n---\n".join(e.content[-600:] for e in fail_evidence[-2:])

        # ===== BUILD PROMPT =====
        user_msg = f"## Function to complete\n```python\n{problem_prompt}\n```"

        if micro_warnings:
            user_msg += f"\n\n{micro_warnings}"

        # Include planner hints on first attempt only
        if plan_hint and not prior_code:
            user_msg += f"\n\n## Pre-flight analysis (use this to guide your implementation)\n{plan_hint}"

        if prior_code:
            user_msg += f"\n\n## Previous attempt (needs refinement)\n```python\n{prior_code}\n```"

        if feedback:
            user_msg += f"\n\n## Reviewer feedback — address ALL directives\n{feedback}"

        if failure_info:
            user_msg += f"\n\n## Test failure output (FIX THESE ISSUES)\n{failure_info}"

        # Diversity hint from branch-and-merge (each branch gets a different directive)
        diversity_hint = task.metadata.get("diversity_hint", "")
        if diversity_hint:
            user_msg += f"\n\n## Branch directive\n{diversity_hint}"

        if len(attempted_hashes) >= 2:
            user_msg += (
                f"\n\n## IMPORTANT: {len(attempted_hashes)} attempts have already failed. "
                "Do NOT repeat a previous solution. Try a fundamentally different approach."
            )

        # ===== CALL LLM =====
        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        # Slightly higher temperature when looping to encourage diversity
        temperature = 0.3 if not prior_code else (0.5 if len(attempted_hashes) >= 3 else 0.2)
        response = await self.call_llm(messages, temperature=temperature)
        code = self.parser.parse(response)

        # ===== POST ARTIFACT UNDER ROOT TASK ID =====
        # KEY FIX: always post under root_task_id so the episode loop can find it.
        # Previously, fix tasks posted code under fix_task.task_id which the
        # episode loop never checked.
        provenance = [prior_code_artifact.artifact_id] if prior_code_artifact else []
        artifact = Artifact(
            type=ArtifactType.CODE,
            task_id=root_task_id,
            author=self.agent_id,
            content=code,
            version=(prior_code_artifact.version + 1) if prior_code_artifact else 1,
            provenance=provenance,
            metadata={"raw_response": response[:500]},
        )

        self.logger.info(
            f"{self.name}: Generated {len(code)} chars of code "
            f"(version {artifact.version}, root_task={root_task_id}, "
            f"plan={'yes' if plan_hint else 'no'}, feedback={'yes' if feedback else 'no'})"
        )
        if micro_warnings:
            self.logger.info(f"{self.name}: Applied micro-checker warnings")

        return [artifact]
