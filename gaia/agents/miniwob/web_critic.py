"""WebCritic agent — reviews action history and gives corrective feedback"""

from typing import List

from ...blackboard.models import Task, Artifact, ArtifactType, SignalType, Signal
from ...agents.base import BaseAgent
from ...prompts.miniwob.critic import WebCriticPrompts


_PROMPTS = WebCriticPrompts()


class WebCriticAgent(BaseAgent):
    """Reviews navigator actions and posts corrective feedback.

    Reads:  CONFLICT signals (task stuck / no progress)
            ACTION artifacts (action history)
            PLAN artifact
            DOCUMENTATION artifact (current DOM)
    Writes: REVIEW artifact with FAILURE_REASON + SUGGESTED_ACTION
    """

    def should_claim_task(self, task: Task) -> bool:
        """Claim web tasks that have a conflict (navigator is stuck)"""
        if task.metadata.get("task_type") != "web_interaction":
            return False
        # Only review when there are conflict signals
        conflicts = [
            s for s in self.blackboard.get_signals(signal_type=SignalType.CONFLICT)
            if s.task_id == task.task_id and not s.resolved
        ]
        return len(conflicts) > 0

    async def execute(self, task: Task) -> List[Artifact]:
        instruction = task.metadata.get("instruction", task.description)
        action_history = task.metadata.get("action_history", [])
        history_text = "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(action_history)
        ) or "(no actions recorded)"

        plan_artifact = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.PLAN)
        plan_text = plan_artifact.content if plan_artifact else "(no plan)"

        # Read DOM elements directly from task metadata (no DOMAnalyzer dependency)
        dom_elements = task.metadata.get("dom_elements", [])
        elements_text = "\n".join(
            f'[{el.get("tag","?")}] "{el.get("text","")}" ref={el.get("ref","")}'
            for el in dom_elements
            if el.get("text") or el.get("tag") in ("button", "input", "input_text", "input_checkbox", "input_radio", "input_number", "select", "a", "textarea", "option", "t")
        ) or "(no DOM info)"

        messages = [
            {"role": "system", "content": _PROMPTS.SYSTEM},
            {
                "role": "user",
                "content": _PROMPTS.REVIEW_ACTIONS.format(
                    instruction=instruction,
                    plan=plan_text,
                    action_history=history_text,
                    elements=elements_text,
                ),
            },
        ]
        feedback = await self.call_llm(messages, temperature=0.3)

        # Resolve all pending CONFLICT signals for this task so Navigator can proceed
        for signal in self.blackboard.get_signals(signal_type=SignalType.CONFLICT):
            if signal.task_id == task.task_id and not signal.resolved:
                self.blackboard.resolve_signal(signal.signal_id)

        self.logger.info(f"{self.name}: Posted critique + resolved conflicts for {task.task_id}")

        return [
            Artifact(
                type=ArtifactType.REVIEW,
                task_id=task.task_id,
                author=self.agent_id,
                content=feedback,
                metadata={"step": task.metadata.get("current_step", 0)},
            )
        ]
