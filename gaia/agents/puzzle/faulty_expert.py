"""FaultyExpertAgent: an ExpertAgent that receives a mix of correct and corrupted clues.

Used in the Fault Injection experiment (E9). Simulates a real-world failure mode
where one agent's data source is unreliable (bad sensor, wrong document,
hallucinating sub-agent). Without detection, this corruption propagates through
the synthesizer and silently produces a wrong solution.

The DeductionAuditorAgent + TrustAwareSynthesizerAgent are designed to detect
and discount this agent's faulty deductions.

Mechanism:
  - At construction, `noise_clues` are pre-generated (contradictory clues
    derived from the puzzle's correct solution — e.g., "Alice's pet is NOT cat"
    when the correct answer is "Alice has a cat").
  - `corruption_rate` fraction of the agent's real clues are replaced with
    noise clues (deterministic, seeded).
  - The agent then runs exactly like ExpertAgent — it reasons over the
    corrupted input as if it were correct.

Output artifact is identical to ExpertAgent (subtype="partial_deduction") so
the auditor and synthesizer can process it uniformly. The metadata flag
"is_faulty": True is for analysis only — the agent itself doesn't know it's
faulty, and the rest of the system shouldn't either (that's the whole point
of the experiment).
"""

import random
from typing import List, Optional
from ...blackboard.models import Task, Artifact, ArtifactType
from ...prompts.puzzle.expert import ExpertPrompts
from .expert import ExpertAgent


class FaultyExpertAgent(ExpertAgent):
    """An ExpertAgent with a fraction of its clues replaced by noise.

    Inherits from ExpertAgent so the rest of the pipeline (Synthesizer,
    Critic, Verifier, Auditor) treats it identically. The episode loop
    routes tasks to it the same way.

    Args:
        partition: "A" or "B" — clue partition assigned to this expert.
        noise_clues: list of contradictory clue strings to inject. Generated
            at experiment setup time from the puzzle's known correct solution.
        corruption_rate: fraction of real clues to replace with noise (0.0–1.0).
            Default 0.3 means 30% of clues become noise.
        seed: RNG seed for reproducible corruption across runs.
    """

    def __init__(
        self,
        partition: str,
        noise_clues: List[str],
        corruption_rate: float = 0.3,
        seed: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("name", f"FaultyExpert-{partition}")
        kwargs.setdefault("role", "faulty_expert")
        super().__init__(partition=partition, **kwargs)
        self.noise_clues = noise_clues
        self.corruption_rate = corruption_rate
        self.rng = random.Random(seed)

    def _corrupt_clues(self, real_clues: List[str]) -> List[str]:
        """Replace a fraction of real clues with noise clues.

        Strategy: pick `n_replace` indices uniformly at random, swap each with
        a randomly chosen noise clue. If noise pool is smaller than n_replace,
        we sample WITH replacement (repeats are fine — they're all wrong anyway).
        """
        if not self.noise_clues or self.corruption_rate <= 0:
            return list(real_clues)

        n_replace = max(1, round(len(real_clues) * self.corruption_rate))
        n_replace = min(n_replace, len(real_clues))

        corrupted = list(real_clues)
        replace_indices = self.rng.sample(range(len(corrupted)), n_replace)

        for idx in replace_indices:
            noise = self.rng.choice(self.noise_clues)
            corrupted[idx] = noise

        return corrupted

    async def execute(self, task: Task) -> List[Artifact]:
        real_clues = task.metadata.get("clues", [])
        total_clues = task.metadata.get("total_clues", len(real_clues))

        corrupted_clues = self._corrupt_clues(real_clues)
        n_corrupted = sum(1 for r, c in zip(real_clues, corrupted_clues) if r != c)

        self.logger.warning(
            f"{self.name}: Running with {n_corrupted}/{len(real_clues)} clues corrupted "
            f"(rate={self.corruption_rate})"
        )

        messages = [
            {"role": "system", "content": self.prompts.SYSTEM},
            {"role": "user", "content": self.prompts.format_user(
                partition=self.partition,
                clues=corrupted_clues,
                total_clues=total_clues,
            )},
        ]

        temperature = task.metadata.get("temperature", 0.2)
        response = await self.call_llm(messages, temperature=temperature)

        root_task_id = task.parent_id or task.task_id

        artifact = Artifact(
            type=ArtifactType.PLAN,
            task_id=root_task_id,
            author=self.agent_id,
            content=response,
            metadata={
                "subtype": "partial_deduction",
                "partition": self.partition,
                "agent_name": self.name,
                "clues": corrupted_clues,
                "real_clues": real_clues,
                "is_faulty": True,
                "corruption_rate": self.corruption_rate,
                "n_corrupted": n_corrupted,
            },
        )

        self.logger.info(
            f"{self.name} (Partition {self.partition}, FAULTY): "
            f"Posted partial deduction with {n_corrupted} corrupted clues"
        )
        return [artifact]
