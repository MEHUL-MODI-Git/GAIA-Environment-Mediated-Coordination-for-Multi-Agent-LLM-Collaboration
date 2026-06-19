"""PuzzleVerifierAgent: Python constraint solver + LLM sanity check."""

import re
from itertools import permutations
from typing import Dict, List, Optional, Tuple
from ...blackboard.models import Task, Artifact, ArtifactType, Evidence
from ...prompts.puzzle.verifier import PuzzleVerifierPrompts
from ..base import BaseAgent

PEOPLE = ["Alice", "Bob", "Carol", "Dave"]
JOBS   = ["doctor", "teacher", "engineer", "artist"]
PETS   = ["cat", "dog", "fish", "bird"]
DRINKS = ["coffee", "tea", "juice", "water"]


# ---------------------------------------------------------------------------
# Python constraint solver (deterministic ground truth)
# ---------------------------------------------------------------------------

def is_consistent(candidate: Dict, structs: List[dict]) -> bool:
    for s in structs:
        t = s["type"]
        if t == "person_attr":
            if candidate[s["person"]][s["attr"]] != s["val"]:
                return False
        elif t == "attr_attr":
            person1 = next((p for p in PEOPLE if candidate[p][s["attr1"]] == s["val1"]), None)
            if person1 is None:
                return False
            if candidate[person1][s["attr2"]] != s["val2"]:
                return False
        elif t == "neg_person_attr":
            if candidate[s["person"]][s["attr"]] == s["val"]:
                return False
    return True


def python_solve(structs: List[dict]) -> List[Dict]:
    """Find all solutions consistent with structured clues (brute force)."""
    found = []
    for jp in permutations(JOBS):
        for pp in permutations(PETS):
            for dp in permutations(DRINKS):
                c = {p: {"job": j, "pet": pet, "drink": d}
                     for p, j, pet, d in zip(PEOPLE, jp, pp, dp)}
                if is_consistent(c, structs):
                    found.append(c)
    return found


def proposed_matches_ground_truth(
    proposed: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, Dict[str, str]],
) -> Tuple[bool, str]:
    """
    Compare proposed solution to ground truth.
    Returns (passed, diff_description).
    """
    diffs = []
    for person in PEOPLE:
        if person not in proposed:
            diffs.append(f"{person}: missing from proposed solution")
            continue
        for attr in ["job", "pet", "drink"]:
            p_val = proposed[person].get(attr, "?")
            g_val = ground_truth[person][attr]
            if p_val != g_val:
                diffs.append(f"{person}.{attr}: proposed={p_val}, correct={g_val}")

    if diffs:
        return False, "; ".join(diffs)
    return True, "All assignments match ground truth"


def parse_solution_from_text(text: str) -> Optional[Dict[str, Dict[str, str]]]:
    """Extract structured solution from free-text."""
    solution = {}
    for person in PEOPLE:
        pattern = (
            rf"{person}\s*:\s*"
            rf"job\s*=\s*(\w+)\s*[,;]?\s*"
            rf"pet\s*=\s*(\w+)\s*[,;]?\s*"
            rf"drink\s*=\s*(\w+)"
        )
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            solution[person] = {
                "job":   m.group(1).strip().lower(),
                "pet":   m.group(2).strip().lower(),
                "drink": m.group(3).strip().lower(),
            }
    return solution if len(solution) == len(PEOPLE) else None


class PuzzleVerifierAgent(BaseAgent):
    """Verifies the synthesized solution using Python constraint solver (ground truth).

    Steps:
    1. Read the latest REVIEW artifact (proposed solution) from blackboard.
    2. Parse its structured solution.
    3. Compare to stored ground truth (deterministic).
    4. Post Evidence(passed=True/False).
    5. Optionally run LLM sanity check for logging.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("role", "puzzle_verifier")
        super().__init__(**kwargs)
        self.prompts = PuzzleVerifierPrompts()

    def should_claim_task(self, task: Task) -> bool:
        return task.metadata.get("task_type") == "verify"

    async def execute(self, task: Task) -> List[Artifact]:
        root_task_id = task.parent_id or task.task_id
        ground_truth: Dict = task.metadata.get("solution", {})
        all_clues_text: List[str] = task.metadata.get("all_clues_text", [])
        all_clues_structs: List[dict] = task.metadata.get("all_clues_structs", [])

        # Find latest proposed solution
        all_artifacts = self.blackboard.get_artifacts_for_task(root_task_id)
        solution_artifacts = [
            a for a in all_artifacts
            if a.type == ArtifactType.REVIEW
            and a.metadata.get("subtype") == "proposed_solution"
        ]

        if not solution_artifacts:
            self.logger.warning(f"{self.name}: No proposed solution found to verify")
            evidence = Evidence(
                type="puzzle_check",
                content="No proposed solution found",
                passed=False,
                metadata={"reason": "no_solution"},
            )
            self.blackboard.post_evidence(evidence)
            return []

        # Use the latest solution (could be reconciled version)
        latest_solution_artifact = solution_artifacts[-1]
        solution_text = latest_solution_artifact.content

        # --- Python ground truth check ---
        # First try parsed_solution from artifact metadata (set by SynthesizerAgent)
        proposed = latest_solution_artifact.metadata.get("parsed_solution")
        if not proposed:
            # Fall back to parsing the text
            proposed = parse_solution_from_text(solution_text)

        if not proposed:
            passed = False
            diff_desc = "Could not parse a structured solution from synthesizer output"
            self.logger.warning(f"{self.name}: Failed to parse solution text")
        else:
            passed, diff_desc = proposed_matches_ground_truth(proposed, ground_truth)

        # --- Also verify proposed solution is consistent with ALL clues ---
        if proposed and all_clues_structs:
            clue_consistent = is_consistent(proposed, all_clues_structs)
            if not clue_consistent:
                passed = False
                diff_desc += " [also fails clue consistency check]"

        self.logger.info(
            f"{self.name}: Python check → {'PASS' if passed else 'FAIL'} | {diff_desc}"
        )

        # --- LLM sanity check (logged but NOT used as ground truth) ---
        llm_verdict = "skipped"
        if all_clues_text and proposed:
            proposed_text = "\n".join(
                f"{p}: job={v['job']}, pet={v['pet']}, drink={v['drink']}"
                for p, v in proposed.items()
            )
            messages = [
                {"role": "system", "content": self.prompts.SYSTEM},
                {"role": "user", "content": self.prompts.format_user(
                    solution_text=proposed_text,
                    all_clues=all_clues_text,
                )},
            ]
            try:
                llm_response = await self.call_llm(messages, temperature=0.0)
                llm_verdict = "PASS" if re.search(r"\bPASS\b", llm_response, re.IGNORECASE) else "FAIL"
            except Exception as e:
                llm_verdict = f"error: {e}"

        # Post evidence to blackboard
        evidence = Evidence(
            type="puzzle_check",
            artifact_id=latest_solution_artifact.artifact_id,
            content=(
                f"Python ground truth: {'PASS' if passed else 'FAIL'}\n"
                f"Diff: {diff_desc}\n"
                f"LLM sanity check: {llm_verdict}\n"
                f"Proposed solution: {proposed}"
            ),
            passed=passed,
            metadata={
                "proposed_solution": proposed,
                "ground_truth": ground_truth,
                "diff": diff_desc,
                "llm_verdict": llm_verdict,
                "python_verdict": "PASS" if passed else "FAIL",
            },
        )
        self.blackboard.post_evidence(evidence)

        return []
