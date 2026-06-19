"""Critic agent - reviews code with structured output protocol"""

from typing import List
from ..blackboard.models import Task, Artifact, ArtifactType, Claim, Signal, SignalType
from ..prompts.critic import CriticPrompts
from .base import BaseAgent


class CriticAgent(BaseAgent):
    """Fast-tier gatekeeper agent that reviews code and produces structured feedback.

    Output format:
    - LGTM: code is correct
    - FAILURE_HYPOTHESIS + FIX_DIRECTIVES + BOUNDARY_TESTS: code has bugs
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "Critic")
        kwargs.setdefault("role", "critic")
        super().__init__(**kwargs)
        self.prompts = CriticPrompts()

    def should_claim_task(self, task: Task) -> bool:
        """Skip tasks that already have a REVIEW artifact.

        Uses root_task_id scope: for fix tasks, check code under the root
        task (where Coder posts it), not the fix task itself.
        """
        root_task_id = task.parent_id or task.task_id
        review_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.REVIEW)
        if review_artifact:
            return False
        code_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.CODE)
        if not code_artifact:
            return False
        return True

    def _is_lgtm(self, text: str) -> bool:
        """Detect if response signals approval.

        Primary check: response must START with 'LGTM' (as required by protocol).
        Fallback: positive phrases with no structured-failure keywords.
        """
        stripped = text.strip()
        # Primary: protocol requires LGTM at the very start
        if stripped.upper().startswith("LGTM"):
            return True
        t = stripped.lower()
        # Definitive negatives — these mean failure regardless of anything else
        hard_negatives = ["failure_hypothesis", "fix_directives", "fix directive"]
        if any(n in t for n in hard_negatives):
            return False
        # Fallback positive signals
        positives = ["looks good", "no issues", "correct implementation",
                     "correctly implements", "passes all", "no bugs found"]
        return any(p in t for p in positives)

    async def execute(self, task: Task) -> List[Artifact]:
        """Review code for the task using structured output protocol.

        Args:
            task: Task with a CODE artifact to review

        Returns:
            List containing REVIEW artifact
        """
        # Code is always posted under root task id (Coder's contract)
        root_task_id = task.parent_id or task.task_id
        code_artifact = self.blackboard.get_latest_artifact(root_task_id, ArtifactType.CODE)
        if not code_artifact:
            self.logger.warning(f"{self.name}: No code artifact found for root {root_task_id}")
            return []

        problem_prompt = task.description
        code = code_artifact.content

        # Get micro-checker warnings (injected by episode loop)
        micro_warnings = task.metadata.get("micro_checker_warnings", "")

        # Get test results if available (evidence is stored on the artifact)
        test_evidence = self.blackboard.get_evidence_for_artifact(code_artifact.artifact_id)
        test_results = ""
        passed = None
        if test_evidence:
            latest_test = test_evidence[0]
            test_results = latest_test.content
            passed = latest_test.passed

        # Build prompt with structured output format
        if test_results:
            prompt = self.prompts.REVIEW_WITH_TEST_RESULTS.format(
                problem_prompt=problem_prompt,
                code=code,
                test_results=test_results,
                passed_str="PASSED" if passed else "FAILED",
                micro_warnings=f"\n{micro_warnings}" if micro_warnings else "",
            )
        else:
            prompt = self.prompts.REVIEW.format(
                problem_prompt=problem_prompt,
                code=code,
                micro_warnings=f"\n{micro_warnings}" if micro_warnings else "",
            )

        # Call LLM at low temperature for consistent structured output
        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = await self.call_llm(messages, temperature=0.1)

        lgtm = self._is_lgtm(response)

        # Post FULL structured response as claim under ROOT task so Coder/Fixer find it
        claim = Claim(
            statement=response[:1500],  # Extended limit so FIX_DIRECTIVES are never truncated
            confidence=0.9 if lgtm else 0.6,
            task_id=root_task_id,
            author=self.agent_id,
            evidence_ids=[e.evidence_id for e in test_evidence] if test_evidence else [],
        )
        self.blackboard.post_claim(claim)

        self.logger.info(
            f"{self.name}: Posted {'LGTM' if lgtm else 'issues found'} claim for task {task.task_id}"
        )

        # Post CONFLICT signal if issues found (episode loop compatibility)
        if not lgtm:
            signal = Signal(
                type=SignalType.CONFLICT,
                task_id=root_task_id,
                description=f"Critic found issues: {response[:100]}",
                severity=0.7,
                metadata={"review_artifact_id": root_task_id, "conflict_type": "review_failure"},
            )
            self.blackboard.post_signal(signal)
            self.logger.info(f"{self.name}: Posted conflict signal for task {root_task_id}")

        # Create REVIEW artifact with full structured response
        artifact = Artifact(
            type=ArtifactType.REVIEW,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={"is_lgtm": lgtm},
        )

        return [artifact]

    async def provide_feedback_after_verification(
        self, task: Task, code: str, test_output: str
    ) -> str:
        """Provide structured feedback after verification failure.

        Called by the episode loop when verification fails. Returns structured
        FIX_DIRECTIVES for the next Coder/Fixer iteration.
        """
        problem_prompt = task.description

        prompt = self.prompts.FEEDBACK_AFTER_VERIFICATION.format(
            problem_prompt=problem_prompt,
            code=code,
            test_output=test_output,
        )

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": prompt},
        ]

        feedback = await self.call_llm(messages, temperature=0.1)

        self.logger.info(
            f"{self.name}: Generated post-verification feedback ({len(feedback)} chars)"
        )

        return feedback
