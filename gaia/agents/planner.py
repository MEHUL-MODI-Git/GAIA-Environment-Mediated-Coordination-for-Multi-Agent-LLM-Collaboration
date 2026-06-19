"""Planner agent - produces strategic hints for the Coder (no subtask decomposition)

Design principles:
- Runs ONCE per episode, before the first Coder iteration
- Only triggers on complex problems (complexity heuristic, score >= 2)
- Posts a PLAN artifact with: algorithm hint, edge cases, approach note
- Does NOT create subtasks on the blackboard (avoids confusing other agents)
- Coder reads the PLAN artifact and uses it for a better first attempt
"""

from typing import List
from ..blackboard.models import Task, Artifact, ArtifactType
from .base import BaseAgent


# ── Complexity heuristic ──────────────────────────────────────────────────────
_COMPLEX_KEYWORDS = [
    "sort", "permut", "combin", "recur", "fibonac", "palindrom",
    "bracket", "parenthes", "prime", "factor", "gcd", "lcm",
    "suffix", "prefix", "subsequence", "substring", "anagram",
    "roman", "binary", "median", "circular", "spiral", "zigzag",
    "matrix", "graph", "tree", "stack", "queue", "dp", "dynamic",
    "minimum", "maximum", "longest", "shortest", "digit", "negative",
    "threshold", "boundary", "unique", "distinct",
]

_PLANNER_SYSTEM = """You are a senior Python engineer doing pre-flight analysis before coding.
Your analysis is consumed by a Coder agent who will write the actual implementation.

Output EXACTLY this format — no extra text:

ALGORITHM:
<1-3 sentences: the cleanest algorithm for this problem>

EDGE_CASES:
- <specific edge case 1 with example input/output>
- <specific edge case 2>
- <add more as needed, at least 3>

BOUNDARY_WATCH:
<1-2 sentences on the trickiest boundary condition: < vs <=, off-by-one, empty list, etc.>

APPROACH_NOTE:
<any non-obvious requirement or known pitfall in the spec>
"""

_PLANNER_PROMPT = """Analyze this Python function before it is implemented:

{problem_prompt}

Output the structured analysis only."""

_MULTI_APPROACH_PROMPT = """A Python coding problem has failed to be solved after multiple attempts with the same approach.

Problem:
{problem_prompt}

What went wrong (latest test failure):
{failure_summary}

Generate exactly {n} DISTINCT algorithmic approaches to solve this problem.
Each must use a fundamentally different strategy — not just a variation of the same idea.
Be concrete and specific: name the data structure, algorithm, or technique, and describe the key implementation steps.

Format your response EXACTLY as:
APPROACH_1: [specific algorithm/approach with key implementation details, 2-3 sentences]
APPROACH_2: [different algorithm/approach with key implementation details, 2-3 sentences]
APPROACH_3: [third distinct algorithm/approach with key implementation details, 2-3 sentences]

Do not include code — just the approach description."""


def _complexity_score(prompt: str) -> int:
    """Return 0-10 complexity score based on problem characteristics."""
    text = prompt.lower()
    score = sum(1 for kw in _COMPLEX_KEYWORDS if kw in text)
    if len(prompt.split()) > 80:
        score += 2
    if prompt.count("assert") >= 3:
        score += 1
    return score


class PlannerAgent(BaseAgent):
    """Slow-tier agent that posts strategic hints for complex problems.

    Runs ONCE at episode start. Posts a PLAN artifact the Coder uses.
    Does NOT decompose into subtasks.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "Planner")
        kwargs.setdefault("role", "planner")
        super().__init__(**kwargs)

    def should_claim_task(self, task: Task) -> bool:
        """Only claim root tasks that have no PLAN artifact yet and are complex."""
        # Never claim subtasks / fix tasks
        if task.parent_id is not None:
            return False
        # Only plan once
        plan = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.PLAN)
        if plan:
            return False
        # Don't plan if Coder already ran (too late)
        code = self.blackboard.get_latest_artifact(task.task_id, ArtifactType.CODE)
        if code:
            return False
        # Skip simple problems
        score = _complexity_score(task.description)
        if score < 2:
            return False
        return True

    async def execute(self, task: Task) -> List[Artifact]:
        """Produce a PLAN artifact with strategic hints."""
        score = _complexity_score(task.description)
        self.logger.info(f"{self.name}: Producing plan (complexity={score})")

        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": _PLANNER_PROMPT.format(
                problem_prompt=task.description
            )},
        ]

        response = await self.call_llm(messages, temperature=0.3)

        artifact = Artifact(
            type=ArtifactType.PLAN,
            task_id=task.task_id,
            author=self.agent_id,
            content=response,
            metadata={"complexity_score": score},
        )

        self.logger.info(f"{self.name}: PLAN posted for {task.task_id}")
        return [artifact]

    async def generate_approaches(
        self, problem_prompt: str, failure_summary: str, n: int = 3
    ) -> list:
        """Generate N distinct algorithmic approaches for a stuck problem.

        Called directly by branch_merge when standard refinement has stalled.
        NOT part of the normal blackboard polling / episode-start flow.

        Returns:
            List of N approach description strings. Falls back to generic hints if parsing fails.
        """
        self.logger.info(f"{self.name}: Generating {n} diverse approaches for branch-and-merge")

        prompt = _MULTI_APPROACH_PROMPT.format(
            problem_prompt=problem_prompt,
            failure_summary=failure_summary,
            n=n,
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = await self.call_llm(messages, temperature=0.4)

        # Parse APPROACH_1: ... APPROACH_2: ... APPROACH_3: ...
        approaches = []
        for i in range(1, n + 1):
            marker = f"APPROACH_{i}:"
            if marker in response:
                start = response.index(marker) + len(marker)
                next_marker = f"APPROACH_{i + 1}:"
                end = response.index(next_marker) if next_marker in response else len(response)
                approaches.append(response[start:end].strip())

        if len(approaches) < n:
            self.logger.warning(
                f"{self.name}: Could only parse {len(approaches)}/{n} approaches, "
                "filling with generic hints"
            )
            generic = [
                "Use a completely different algorithm than the obvious approach. Think about the problem from scratch.",
                "Focus especially on boundary and edge cases: empty inputs, single elements, negative numbers, zero.",
                "Prioritize correctness over cleverness. Use simple loops and explicit conditions rather than one-liners.",
            ]
            while len(approaches) < n:
                approaches.append(generic[len(approaches) % len(generic)])

        self.logger.info(f"{self.name}: Generated {len(approaches)} approaches")
        return approaches
