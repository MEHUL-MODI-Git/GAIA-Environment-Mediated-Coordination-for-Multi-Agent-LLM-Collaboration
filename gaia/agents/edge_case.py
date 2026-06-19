"""Fixer agent - applies FIX_DIRECTIVES from Critic to fix failing code.

Repurposed from EdgeCaseAgent. Instead of generic edge case analysis,
this agent reads the Critic's structured FIX_DIRECTIVES and applies
targeted, precise fixes to the failing code.
"""

from typing import List
from ..blackboard.models import Task, Artifact, ArtifactType
from ..parsers.code_parser import CodeParser
from .base import BaseAgent


FIXER_SYSTEM = """You are a debugging expert in a multi-agent team.
The Critic has diagnosed the bug and provided FIX_DIRECTIVES.
Your job is to apply those directives precisely to fix the failing code.

Rules:
- Do NOT refactor or restructure the code — only fix the identified bugs
- Apply EVERY directive from FIX_DIRECTIVES
- Ensure the BOUNDARY_TESTS from the Critic would pass
- Write the complete corrected function wrapped in ```python ... ``` markers
- Include any needed imports at the top
"""


class EdgeCaseAgent(BaseAgent):
    """Fast-tier Fixer agent: applies Critic's FIX_DIRECTIVES for targeted repair.

    Activated when:
    - Verifier reports test failures (via conflict resolution in episode loop)
    - On 3+ repeated failures (same as before)

    Key difference from old EdgeCase: reads structured FIX_DIRECTIVES from Critic
    claims instead of doing generic edge case diagnosis.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "Fixer")
        kwargs.setdefault("role", "edge_case")
        super().__init__(**kwargs)
        self.parser = CodeParser()

    async def execute(self, task: Task) -> List[Artifact]:
        """Apply FIX_DIRECTIVES to fix the failing code.

        Args:
            task: Task with failure metadata. May be a fix task (has parent_id)
                  or root task with edge_case_analysis metadata.

        Returns:
            List containing fixed CODE artifact posted under root task id.
        """
        # Determine root task scope
        root_task_id = task.parent_id or task.task_id

        # Get latest failing code (from root scope)
        code_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.CODE)
        code = ""
        if code_artifact:
            code = code_artifact.content
        else:
            # Fallback to metadata
            code = task.metadata.get("previous_code", "")

        # Get the most recent FAILURE claim from Critic (posted under root task)
        claims = self.blackboard.storage.get_claims_for_task(root_task_id)
        fix_directives = ""
        for c in reversed(claims):
            stmt_lower = c.statement.lower()
            if "fix_directives" in stmt_lower or "failure_hypothesis" in stmt_lower:
                raw = c.statement
                # Extract FAILURE_HYPOTHESIS + FIX_DIRECTIVES cleanly
                # Strip BOUNDARY_TESTS/RISK_FLAGS which are noise for the fixer
                if "FIX_DIRECTIVES:" in raw:
                    hyp_start = raw.find("FAILURE_HYPOTHESIS:")
                    dir_start = raw.find("FIX_DIRECTIVES:")
                    dir_section = raw[dir_start:]
                    for cutoff in ["BOUNDARY_TESTS:", "RISK_FLAGS:"]:
                        if cutoff in dir_section:
                            dir_section = dir_section[:dir_section.index(cutoff)]
                    if hyp_start >= 0:
                        fix_directives = raw[hyp_start:dir_start] + dir_section
                    else:
                        fix_directives = dir_section
                else:
                    fix_directives = raw
                break

        # Get test failure info from task metadata
        failure_info = task.metadata.get("feedback", "")
        problem_prompt = task.description

        # Also get test failure evidence if available
        test_failure_content = ""
        if code_artifact:
            fail_evidence = [
                e for e in self.blackboard.get_evidence_for_artifact(code_artifact.artifact_id)
                if e.passed is False
            ]
            if fail_evidence:
                test_failure_content = fail_evidence[-1].content[-600:]

        # Build targeted repair prompt
        user_msg = f"## Spec\n{problem_prompt}\n\n## Failing code\n```python\n{code}\n```"

        if fix_directives:
            user_msg += f"\n\n## Critic's diagnosis and fix directives\n{fix_directives}"
        elif failure_info:
            user_msg += f"\n\n## Failure information\n{failure_info}"

        if test_failure_content:
            user_msg += f"\n\n## Test failure output\n{test_failure_content}"

        messages = [
            {"role": "system", "content": FIXER_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        response = await self.call_llm(messages, temperature=0.4)
        fixed_code = self.parser.parse(response)

        self.logger.info(
            f"{self.name}: Generated fix ({len(fixed_code)} chars) "
            f"{'using FIX_DIRECTIVES' if fix_directives else 'from failure info'}"
        )

        # Post under root task id so episode loop can find it
        artifact = Artifact(
            type=ArtifactType.CODE,
            task_id=root_task_id,
            author=self.agent_id,
            content=fixed_code,
            version=(code_artifact.version + 1) if code_artifact else 1,
            provenance=[code_artifact.artifact_id] if code_artifact else [],
            metadata={"fix_type": "targeted_repair", "had_directives": bool(fix_directives)},
        )

        return [artifact]
