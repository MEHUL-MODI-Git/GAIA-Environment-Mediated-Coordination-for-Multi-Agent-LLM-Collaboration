"""HumanEval dataset loader"""

import json
from pathlib import Path
from typing import List, Dict, Any


class HumanEvalLoader:
    """Load HumanEval problems from test.jsonl"""

    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to test.jsonl file
        """
        self.data_path = Path(data_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load all HumanEval problems

        Returns:
            List of problem dicts with keys:
            - task_id: e.g. "HumanEval/0"
            - prompt: Function signature + docstring
            - test: Test harness with check() function
            - entry_point: Function name
            - canonical_solution: Reference solution (for analysis)
        """
        problems = []

        with open(self.data_path) as f:
            for line in f:
                problem = json.loads(line)
                problems.append(problem)

        return problems

    def load_range(self, start: int, end: int) -> List[Dict[str, Any]]:
        """Load a range of problems

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            List of problems in range
        """
        all_problems = self.load()
        return all_problems[start:end]

    def load_by_ids(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """Load specific problems by task_id

        Args:
            task_ids: List of task IDs like ["HumanEval/0", "HumanEval/5"]

        Returns:
            List of matching problems
        """
        all_problems = self.load()
        task_id_set = set(task_ids)
        return [p for p in all_problems if p["task_id"] in task_id_set]

    def count(self) -> int:
        """Get total number of problems"""
        return len(self.load())
