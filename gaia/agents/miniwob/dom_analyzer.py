"""DOMAnalyzer agent — parses DOM observations into structured element lists"""

import json
from typing import List

from ...blackboard.models import Task, Artifact, ArtifactType
from ...agents.base import BaseAgent
from ...prompts.miniwob.dom_analyzer import DOMAnalyzerPrompts


_PROMPTS = DOMAnalyzerPrompts()
_MAX_HTML_CHARS = 4000


class DOMAnalyzerAgent(BaseAgent):
    """Parses raw DOM HTML into structured element descriptions.

    Reads:  task metadata (raw_html from latest env step)
    Writes: DOCUMENTATION artifact with structured element list
    """

    def should_claim_task(self, task: Task) -> bool:
        """Claim web tasks that have new raw HTML to parse"""
        if task.metadata.get("task_type") != "web_interaction":
            return False
        # Only analyze if there's fresh HTML in metadata
        return bool(task.metadata.get("raw_html"))

    async def execute(self, task: Task) -> List[Artifact]:
        instruction = task.metadata.get("instruction", task.description)
        raw_html = task.metadata.get("raw_html", "")

        # Truncate HTML to avoid token overload
        truncated_html = raw_html[:_MAX_HTML_CHARS]
        if len(raw_html) > _MAX_HTML_CHARS:
            truncated_html += "\n... [truncated]"

        # If no raw HTML, use structured elements from metadata
        if not raw_html:
            elements_from_meta = task.metadata.get("dom_elements", [])
            if elements_from_meta:
                # Format structured elements directly without LLM
                lines = []
                for el in elements_from_meta:
                    tag = el.get("tag", "?")
                    text = el.get("text", "").strip()
                    ref = el.get("ref", "")
                    if text or tag in ("button", "input", "select", "a"):
                        lines.append(f'[{tag}] "{text}" ref={ref}')
                content = "\n".join(lines) or "(no interactive elements found)"
                return [
                    Artifact(
                        type=ArtifactType.DOCUMENTATION,
                        task_id=task.task_id,
                        author=self.agent_id,
                        content=content,
                        metadata={"source": "structured_elements"},
                    )
                ]
            return []

        messages = [
            {"role": "system", "content": _PROMPTS.SYSTEM},
            {
                "role": "user",
                "content": _PROMPTS.PARSE_DOM.format(
                    instruction=instruction,
                    raw_html=truncated_html,
                    max_chars=_MAX_HTML_CHARS,
                ),
            },
        ]
        parsed = await self.call_llm(messages, temperature=0.0)

        self.logger.info(f"{self.name}: Parsed DOM for {task.task_id}")

        return [
            Artifact(
                type=ArtifactType.DOCUMENTATION,
                task_id=task.task_id,
                author=self.agent_id,
                content=parsed,
                metadata={"source": "llm_dom_parse"},
            )
        ]
