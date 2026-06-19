"""WebPlanner agent — interprets task instruction and produces action plan"""

from typing import List

from ...blackboard.models import Task, Artifact, ArtifactType
from ...agents.base import BaseAgent
from ...prompts.miniwob.planner import WebPlannerPrompts


_PROMPTS = WebPlannerPrompts()


class WebPlannerAgent(BaseAgent):
    """Runs once per episode to produce a high-level action plan.

    Reads:  task instruction + initial DOM observation artifact
    Writes: PLAN artifact consumed by WebNavigatorAgent
    """

    def should_claim_task(self, task: Task) -> bool:
        """Claim web tasks that need a plan.

        Fires on episode start (no plan yet) OR after a strategy reset
        (episode loop clears plan so we regenerate with failure context).
        """
        if task.metadata.get("task_type") != "web_interaction":
            return False
        existing_plan = self.blackboard.get_latest_artifact(
            task.task_id, ArtifactType.PLAN
        )
        if existing_plan is None:
            return True  # No plan yet — initial planning
        # Strategy reset: plan exists but loop flagged a reset is needed
        if task.metadata.get("strategy_reset"):
            # Check that the existing plan hasn't already been superseded
            if existing_plan.metadata.get("superseded"):
                return True
        return False

    async def execute(self, task: Task) -> List[Artifact]:
        instruction = task.metadata.get("instruction", task.description)

        # Read DOM elements from task metadata for context
        dom_elements = task.metadata.get("dom_elements", [])
        elements_text = "\n".join(
            f'[{el.get("tag","?")}] "{el.get("text","")}" ref={el.get("ref","")}'
            for el in dom_elements
            if el.get("text") or el.get("tag") in ("button", "input", "input_text", "input_checkbox", "input_radio", "input_number", "select", "a", "textarea", "option", "t")
        ) or "(initial page — use instruction to plan)"

        # Check if this is a replan (failure feedback available)
        failure_summary = task.metadata.get("failure_summary", "")
        if failure_summary:
            prompt = _PROMPTS.REPLAN.format(
                instruction=instruction,
                failure_summary=failure_summary,
                elements=elements_text,
            )
        else:
            prompt = _PROMPTS.INITIAL.format(
                instruction=instruction,
                elements=elements_text,
            )

        messages = [
            {"role": "system", "content": _PROMPTS.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        is_replan = bool(failure_summary)
        plan_text = await self.call_llm(messages, temperature=0.4 if is_replan else 0.2)

        # Clear the strategy_reset flag now that we've replanned
        if task.metadata.get("strategy_reset"):
            task.metadata["strategy_reset"] = False
            task.metadata["failure_summary"] = ""
            self.blackboard.update_task(task)

        self.logger.info(
            f"{self.name}: {'Re-planned' if is_replan else 'Planned'} for {task.task_id}"
        )

        return [
            Artifact(
                type=ArtifactType.PLAN,
                task_id=task.task_id,
                author=self.agent_id,
                content=plan_text,
                metadata={"instruction": instruction},
            )
        ]
