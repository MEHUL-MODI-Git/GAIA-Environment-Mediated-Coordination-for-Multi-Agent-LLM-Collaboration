"""MiniWoB++ task loader"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class MiniWoBLoader:
    """Load MiniWoB++ task definitions from a JSON file

    Task format (data/miniwob/tasks.json):
    [
      {
        "task_id": "miniwob/click-button",
        "task_name": "click-button",         # gymnasium env name suffix
        "instruction": "Click the Submit button",
        "max_steps": 10,
        "difficulty": "easy",
        "tags": ["click"]
      },
      ...
    ]
    """

    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to tasks.json file
        """
        self.data_path = Path(data_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load all MiniWoB++ tasks

        Returns:
            List of task dicts with keys:
            - task_id: e.g. "miniwob/click-button"
            - task_name: gymnasium env name, e.g. "click-button"
            - instruction: Natural language instruction
            - max_steps: Max steps allowed
            - difficulty: "easy", "medium", "hard"
            - tags: List of capability tags
        """
        with open(self.data_path) as f:
            tasks = json.load(f)
        return tasks

    def load_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Load tasks filtered by difficulty level"""
        return [t for t in self.load() if t.get("difficulty") == difficulty]

    def load_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Load tasks that match any of the given tags"""
        tag_set = set(tags)
        return [t for t in self.load() if set(t.get("tags", [])) & tag_set]

    def load_by_ids(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """Load specific tasks by task_id"""
        task_id_set = set(task_ids)
        return [t for t in self.load() if t["task_id"] in task_id_set]

    def load_by_names(self, task_names: List[str]) -> List[Dict[str, Any]]:
        """Load tasks by task_name (gymnasium env suffix)"""
        name_set = set(task_names)
        return [t for t in self.load() if t.get("task_name") in name_set]

    def count(self) -> int:
        """Get total number of tasks"""
        return len(self.load())
