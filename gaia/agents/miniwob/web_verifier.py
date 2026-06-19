"""WebVerifier agent — checks success_flag from environment and posts evidence"""

from typing import List

from ...blackboard.models import (
    Task,
    Artifact,
    ArtifactType,
    Evidence,
    Signal,
    SignalType,
)
from ...agents.base import BaseAgent


class WebVerifierAgent(BaseAgent):
    """Checks whether the MiniWoB++ task was completed successfully.

    Reads:  task metadata: success_flag, current_step, max_steps
    Writes: EVIDENCE artifact (pass/fail)
            CONFLICT signal if task is stuck (no success within threshold)
    """

    # Raise a conflict signal if no success after this many steps
    CONFLICT_STEP_THRESHOLD = 5

    def should_claim_task(self, task: Task) -> bool:
        """Claim web tasks that have made at least one step"""
        if task.metadata.get("task_type") != "web_interaction":
            return False
        return task.metadata.get("current_step", 0) > 0

    async def execute(self, task: Task) -> List[Artifact]:
        success = task.metadata.get("success_flag", False)
        current_step = task.metadata.get("current_step", 0)
        max_steps = task.metadata.get("max_steps", 15)
        interaction_log = task.metadata.get("interaction_log", "")

        evidence = Evidence(
            type="web_task_result",
            content=interaction_log or f"Step {current_step}/{max_steps}. Success: {success}",
            passed=success,
            metadata={
                "step": current_step,
                "max_steps": max_steps,
                "task_id": task.task_id,
            },
        )
        self.blackboard.post_evidence(evidence)

        if success:
            self.logger.info(f"{self.name}: Task {task.task_id} PASSED at step {current_step}")
            self.blackboard.complete_task(task.task_id, [])
        else:
            # Post conflict signal if stuck for long enough
            if current_step >= self.CONFLICT_STEP_THRESHOLD:
                signal = Signal(
                    type=SignalType.CONFLICT,
                    task_id=task.task_id,
                    description=f"Web task not solved after {current_step} steps",
                    severity=0.7,
                    metadata={
                        "evidence_id": evidence.evidence_id,
                        "step": current_step,
                    },
                )
                self.blackboard.post_signal(signal)
                self.logger.info(
                    f"{self.name}: Posted CONFLICT for {task.task_id} at step {current_step}"
                )
            else:
                self.logger.info(
                    f"{self.name}: Task {task.task_id} not yet done (step {current_step})"
                )

        return [
            Artifact(
                type=ArtifactType.TEST_RESULT,
                task_id=task.task_id,
                author=self.agent_id,
                content=f"success={success} step={current_step}/{max_steps}",
                metadata={"passed": success, "evidence_id": evidence.evidence_id},
            )
        ]
