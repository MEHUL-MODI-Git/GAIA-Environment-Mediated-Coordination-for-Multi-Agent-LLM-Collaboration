"""Parser for extracting task decomposition plans from planner output"""

import re
from typing import List, Dict, Optional
from pydantic import BaseModel


class SubtaskItem(BaseModel):
    """A single subtask from a plan"""

    title: str
    description: str = ""
    priority: float = 1.0
    deps: List[str] = []


class PlanParser:
    """Extract subtask lists from planner output"""

    @staticmethod
    def parse(text: str) -> List[SubtaskItem]:
        """Parse planner output into subtasks

        Expected formats:
        1. Numbered list: "1. Task title\n   Description..."
        2. Bullet list: "- Task title\n  Description..."
        3. Headers: "## Task title\nDescription..."

        Args:
            text: Planner output text

        Returns:
            List of SubtaskItem objects
        """
        subtasks = []

        # Try numbered list format first (most common)
        numbered_pattern = r"(\d+)\.\s+(.+?)(?=\n\d+\.|\Z)"
        matches = re.findall(numbered_pattern, text, re.DOTALL)

        if matches:
            for num, content in matches:
                lines = content.strip().split('\n')
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

                subtasks.append(SubtaskItem(
                    title=title,
                    description=description,
                    priority=float(num),  # Use number as priority
                ))
        else:
            # Try bullet list format
            bullet_pattern = r"[-*]\s+(.+?)(?=\n[-*]|\Z)"
            matches = re.findall(bullet_pattern, text, re.DOTALL)

            for i, content in enumerate(matches):
                lines = content.strip().split('\n')
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

                subtasks.append(SubtaskItem(
                    title=title,
                    description=description,
                    priority=float(len(matches) - i),  # Reverse priority
                ))

        return subtasks

    @staticmethod
    def extract_dependencies(text: str) -> Dict[str, List[str]]:
        """Extract dependency information from plan

        Looks for patterns like:
        - "depends on: task1, task2"
        - "after: task1"
        - "requires: task1"

        Args:
            text: Plan text

        Returns:
            Dict mapping task titles to dependency lists
        """
        deps_map = {}

        # Pattern: "Task X (depends on: Y, Z)"
        pattern = r"(.+?)\s*\((?:depends on|after|requires):\s*(.+?)\)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for task, deps_str in matches:
            task = task.strip()
            deps = [d.strip() for d in deps_str.split(',')]
            deps_map[task] = deps

        return deps_map
