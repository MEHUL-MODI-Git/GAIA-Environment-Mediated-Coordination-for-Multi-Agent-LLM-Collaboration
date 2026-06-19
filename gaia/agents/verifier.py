"""Verifier agent - runs tests and creates evidence"""

from typing import List
from ..blackboard.models import Task, Artifact, ArtifactType, Evidence, Signal, SignalType
from ..execution.code_runner import CodeRunner
from .base import BaseAgent


class VerifierAgent(BaseAgent):
    """Slow-tier agent that runs objective verification"""

    def __init__(self, code_runner: CodeRunner = None, **kwargs):
        kwargs.setdefault("name", "Verifier")
        kwargs.setdefault("role", "verifier")
        super().__init__(**kwargs)
        self.code_runner = code_runner or CodeRunner()

    async def execute(self, task: Task) -> List[Artifact]:
        """Run verification tests for the task

        Args:
            task: Task to verify (should have code artifact and test metadata)

        Returns:
            List of artifacts (usually empty - verification creates Evidence, not Artifacts)
        """
        # Get the latest code artifact
        code_artifact = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.CODE)

        if not code_artifact:
            self.logger.warning(f"{self.name}: No code artifact found for task {task.task_id}")
            return []

        # Get test harness from task metadata
        test = task.metadata.get("test", "")
        entry_point = task.metadata.get("entry_point", "")

        if not test or not entry_point:
            self.logger.warning(f"{self.name}: No test metadata for task {task.task_id}")
            return []

        # Run tests
        passed, test_output = await self.code_runner.run_humaneval_test(
            code_artifact.content, test, entry_point
        )

        # Create evidence
        evidence = Evidence(
            type="test_result",
            content=test_output,
            artifact_id=code_artifact.artifact_id,
            passed=passed,
            metadata={
                "entry_point": entry_point,
                "test_length": len(test),
                "code_length": len(code_artifact.content),
            },
        )

        self.blackboard.post_evidence(evidence)

        self.logger.info(
            f"{self.name}: Verification {'PASSED' if passed else 'FAILED'} "
            f"for task {task.task_id}"
        )

        # Handle test results
        if passed:
            # Tests passed - complete the task!
            self.blackboard.complete_task(task.task_id, [code_artifact])
            self.logger.info(f"{self.name}: ✓ Tests passed! Task {task.task_id} completed")
        else:
            # Tests failed - create conflict signal (leave task CLAIMED for fixes)
            signal = Signal(
                type=SignalType.CONFLICT,
                task_id=task.task_id,
                description=f"Tests failed: {test_output[:100]}",
                severity=0.9,  # High severity for test failures
                metadata={"evidence_id": evidence.evidence_id},
            )
            self.blackboard.post_signal(signal)
            self.logger.info(f"{self.name}: ✗ Posted conflict signal for failed tests")

        # Verifier doesn't create artifacts, only evidence
        return []
