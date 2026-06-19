"""WebNavigator agent — decides and posts the next browser action"""

import json
import re
from typing import List, Optional

from ...blackboard.models import Task, Artifact, ArtifactType, SignalType
from ...agents.base import BaseAgent
from ...prompts.miniwob.navigator import WebNavigatorPrompts


_PROMPTS = WebNavigatorPrompts()

# Artifact type for actions (reuse CODE slot — content is JSON action dict)
_ACTION_TYPE = ArtifactType.CODE


def _parse_action(response: str) -> Optional[dict]:
    """Parse LLM response into a WebAction dict.

    Scans all lines for the first matching action pattern so that
    explanatory text before the action doesn't cause parse failures.

    Handles:
      CLICK: <element>
      FOCUS: <element>            (alias for CLICK)
      TYPE: <text> INTO: <element>
      SELECT: <value> FROM: <element>
      PRESS_KEY: <key>
      DONE
    """
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"DONE[:\s]?$", line, re.IGNORECASE):
            return {"type": "NONE"}

        # CLICK / FOCUS (focus = click to activate element)
        m = re.match(r"(?:CLICK|FOCUS)[:\s]+(.+)", line, re.IGNORECASE)
        if m:
            return {"type": "CLICK", "element_desc": m.group(1).strip()}

        # TYPE ... INTO ...
        m = re.match(r"TYPE:\s*(.+?)\s+INTO:\s*(.+)", line, re.IGNORECASE)
        if m:
            text = m.group(1).strip().strip("\"'")
            return {"type": "TYPE", "text": text, "element_desc": m.group(2).strip()}

        # SELECT ... FROM ...
        m = re.match(r"SELECT:\s*(.+?)\s+FROM:\s*(.+)", line, re.IGNORECASE)
        if m:
            return {"type": "SELECT", "value": m.group(1).strip(), "element_desc": m.group(2).strip()}

        # PRESS_KEY
        m = re.match(r"PRESS_KEY:\s*(\S+)", line, re.IGNORECASE)
        if m:
            return {"type": "PRESS_KEY", "key": m.group(1).strip()}

    return {"type": "UNKNOWN", "raw": response[:200]}


class WebNavigatorAgent(BaseAgent):
    """Decides the next browser action for the current page state.

    Reads:  PLAN artifact (from WebPlannerAgent)
            DOCUMENTATION artifact (from DOMAnalyzerAgent — structured elements)
            REVIEW artifact (from WebCriticAgent — feedback if task failing)
            task metadata: step, max_steps, action_history, instruction
    Writes: CODE artifact containing JSON action dict
    """

    def should_claim_task(self, task: Task) -> bool:
        """Claim web tasks that need a next action.

        Backs off when unresolved CONFLICT signals exist — lets WebCritic
        run first so its feedback is available before the next action.
        """
        if task.metadata.get("task_type") != "web_interaction":
            return False
        if task.metadata.get("success"):
            return False
        # Step aside so Critic can post feedback on this iteration
        unresolved_conflicts = [
            s for s in self.blackboard.get_signals(signal_type=SignalType.CONFLICT)
            if s.task_id == task.task_id and not s.resolved
        ]
        if unresolved_conflicts:
            self.logger.info(f"{self.name}: Backing off — {len(unresolved_conflicts)} conflict(s) pending Critic")
            return False
        return True

    async def execute(self, task: Task) -> List[Artifact]:
        instruction = task.metadata.get("instruction", task.description)
        step = task.metadata.get("current_step", 0)
        max_steps = task.metadata.get("max_steps", 15)
        action_history = task.metadata.get("action_history", [])

        # Format action history
        history_text = "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(action_history)
        ) or "(no actions taken yet)"

        # Read plan
        plan_artifact = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.PLAN)
        plan_text = plan_artifact.content if plan_artifact else "(no plan available)"

        # Read structured DOM elements directly from task metadata
        # (set by episode loop after each browser step — no DOMAnalyzer LLM call needed)
        _INTERACTIVE_TAGS = {
            "button", "input", "input_text", "input_checkbox", "input_radio",
            "input_number", "input_date", "select", "a", "textarea", "option", "t",
        }
        dom_elements = task.metadata.get("dom_elements", [])
        if dom_elements:
            def _fmt_el(el):
                classes = el.get("classes", "").strip()
                suffix = f" [{classes}]" if classes else ""
                return f'[{el.get("tag","?")}] "{el.get("text","")}" ref={el.get("ref","")}{suffix}'
            elements_text = "\n".join(
                _fmt_el(el)
                for el in dom_elements
                if el.get("text") or el.get("tag") in _INTERACTIVE_TAGS
            ) or "(no interactive elements visible)"
        else:
            elements_text = "(page not yet loaded)"

        # Check for critic feedback
        critic_artifact = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.REVIEW)

        if critic_artifact and action_history:
            # Retry with critic feedback
            prompt = _PROMPTS.RETRY_ACTION.format(
                instruction=instruction,
                critic_feedback=critic_artifact.content,
                step=step,
                max_steps=max_steps,
                elements=elements_text,
                action_history=history_text,
            )
        else:
            prompt = _PROMPTS.DECIDE_ACTION.format(
                instruction=instruction,
                plan=plan_text,
                step=step,
                max_steps=max_steps,
                elements=elements_text,
                action_history=history_text,
            )

        messages = [
            {"role": "system", "content": _PROMPTS.SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = await self.call_llm(messages, temperature=0.3)
        action = _parse_action(response)

        self.logger.info(
            f"{self.name}: Step {step} → {action.get('type')} {action.get('element_desc', '')}"
        )

        return [
            Artifact(
                type=_ACTION_TYPE,
                task_id=task.task_id,
                author=self.agent_id,
                content=json.dumps(action),
                metadata={"step": step, "raw_response": response},
            )
        ]
