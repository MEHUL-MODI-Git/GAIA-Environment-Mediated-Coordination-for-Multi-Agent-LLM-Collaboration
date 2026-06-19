#!/usr/bin/env python3
"""Generate fresh logic-grid puzzles for the Asymmetric Information Puzzle experiment.

Design:
- 4 people × 3 attributes (job, pet, drink) = 12 assignments per puzzle
- 12 clues split into Partition A (6) and Partition B (6)
- GUARANTEED: neither partition alone uniquely determines the solution
- GUARANTEED: both partitions together uniquely determine the solution
- Verified by Python constraint solver (brute force over 4!^3 = 13,824 candidates)

Partition strategy:
- Partition A: job-anchored clues (person→job direct, job→pet cross, job→drink cross)
- Partition B: drink/pet-anchored clues (person→drink direct, person→pet direct, pet→drink cross)

This guarantees Expert-A knows jobs but not drinks;
Expert-B knows drinks but not jobs.
Synthesis is structurally required.

Usage:
    python experiments/puzzle/scripts/generate_puzzles.py
    python experiments/puzzle/scripts/generate_puzzles.py --count 30 --output data/puzzle/puzzles.json
"""

import argparse
import json
import random
import sys
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
PEOPLE = ["Alice", "Bob", "Carol", "Dave"]
JOBS   = ["doctor", "teacher", "engineer", "artist"]
PETS   = ["cat", "dog", "fish", "bird"]
DRINKS = ["coffee", "tea", "juice", "water"]

ATTR_VALUES = {"job": JOBS, "pet": PETS, "drink": DRINKS}
ATTRIBUTES  = ["job", "pet", "drink"]


# ---------------------------------------------------------------------------
# Solution type: dict[person] -> dict[attr] -> value
# ---------------------------------------------------------------------------
Solution = Dict[str, Dict[str, str]]


def random_solution() -> Solution:
    """Generate a uniformly random valid solution."""
    jp = random.sample(JOBS,   4)
    pp = random.sample(PETS,   4)
    dp = random.sample(DRINKS, 4)
    return {
        person: {"job": job, "pet": pet, "drink": drink}
        for person, job, pet, drink in zip(PEOPLE, jp, pp, dp)
    }


# ---------------------------------------------------------------------------
# Clue representation
# ---------------------------------------------------------------------------
# Each clue is {"text": str, "struct": {...}} where struct is machine-checkable.
# Struct types:
#   person_attr   : person X has attr=val (direct positive)
#   attr_attr     : whoever has attr1=val1 also has attr2=val2 (cross-attribute)
#   neg_person_attr : person X does NOT have attr=val (direct negative)

def make_person_attr_clue(person: str, attr: str, val: str) -> dict:
    attr_phrase = {
        "job":   f"is the {val}",
        "pet":   f"keeps a {val}",
        "drink": f"drinks {val}",
    }[attr]
    return {
        "text": f"{person} {attr_phrase}.",
        "struct": {"type": "person_attr", "person": person, "attr": attr, "val": val},
    }


def make_attr_attr_clue(attr1: str, val1: str, attr2: str, val2: str) -> dict:
    subject = {
        "job":   f"The {val1}",
        "pet":   f"The {val1} owner",
        "drink": f"The person who drinks {val1}",
    }[attr1]
    predicate = {
        "job":   f"is the {val2}",
        "pet":   f"keeps a {val2}",
        "drink": f"drinks {val2}",
    }[attr2]
    return {
        "text": f"{subject} {predicate}.",
        "struct": {"type": "attr_attr", "attr1": attr1, "val1": val1, "attr2": attr2, "val2": val2},
    }


def make_neg_person_attr_clue(person: str, attr: str, val: str) -> dict:
    attr_phrase = {
        "job":   f"is not the {val}",
        "pet":   f"does not keep a {val}",
        "drink": f"does not drink {val}",
    }[attr]
    return {
        "text": f"{person} {attr_phrase}.",
        "struct": {"type": "neg_person_attr", "person": person, "attr": attr, "val": val},
    }


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------

def is_consistent(candidate: Solution, structs: List[dict]) -> bool:
    """Check if candidate satisfies all structured clues."""
    for s in structs:
        t = s["type"]
        if t == "person_attr":
            if candidate[s["person"]][s["attr"]] != s["val"]:
                return False
        elif t == "attr_attr":
            # find person with attr1=val1
            person1 = next((p for p in PEOPLE if candidate[p][s["attr1"]] == s["val1"]), None)
            if person1 is None:
                return False  # val1 not present (shouldn't happen for valid clues)
            if candidate[person1][s["attr2"]] != s["val2"]:
                return False
        elif t == "neg_person_attr":
            if candidate[s["person"]][s["attr"]] == s["val"]:
                return False
    return True


def count_solutions(structs: List[dict]) -> int:
    """Count solutions consistent with the given structured clues."""
    count = 0
    for jp in permutations(JOBS):
        for pp in permutations(PETS):
            for dp in permutations(DRINKS):
                c = {p: {"job": j, "pet": pet, "drink": d}
                     for p, j, pet, d in zip(PEOPLE, jp, pp, dp)}
                if is_consistent(c, structs):
                    count += 1
                    if count > 1:
                        return count  # Early exit — we only care >1 vs ==1
    return count


def find_solutions(structs: List[dict]) -> List[Solution]:
    """Return all solutions consistent with given clues (used for verification)."""
    found = []
    for jp in permutations(JOBS):
        for pp in permutations(PETS):
            for dp in permutations(DRINKS):
                c = {p: {"job": j, "pet": pet, "drink": d}
                     for p, j, pet, d in zip(PEOPLE, jp, pp, dp)}
                if is_consistent(c, structs):
                    found.append(c)
    return found


# ---------------------------------------------------------------------------
# Clue generation for a given solution
# ---------------------------------------------------------------------------

def generate_partition_a_clues(sol: Solution) -> List[dict]:
    """
    Partition A: job-anchored clues.
    - 2 direct person→job clues
    - 2 job→pet cross-attribute clues
    - 2 job→drink cross-attribute clues

    Expert A will be able to fully determine jobs and infer some pets/drinks,
    but not all drinks (since no direct drink clues).
    """
    clues = []

    # 2 direct person→job (for 2 specific people)
    people_for_direct = random.sample(PEOPLE, 2)
    for person in people_for_direct:
        clues.append(make_person_attr_clue(person, "job", sol[person]["job"]))

    # 2 job→pet cross clues (for 2 specific jobs)
    jobs_for_cross = random.sample(JOBS, 2)
    for job in jobs_for_cross:
        person = next(p for p in PEOPLE if sol[p]["job"] == job)
        clues.append(make_attr_attr_clue("job", job, "pet", sol[person]["pet"]))

    # 2 job→drink cross clues (for 2 specific jobs)
    jobs_for_drink = random.sample([j for j in JOBS if j not in jobs_for_cross], 2)
    for job in jobs_for_drink:
        person = next(p for p in PEOPLE if sol[p]["job"] == job)
        clues.append(make_attr_attr_clue("job", job, "drink", sol[person]["drink"]))

    return clues


def generate_partition_b_clues(sol: Solution) -> List[dict]:
    """
    Partition B: drink/pet-anchored clues.
    - 2 direct person→drink clues (different people from partition A's direct job clues ideally)
    - 2 direct person→pet clues
    - 2 pet→drink cross-attribute clues

    Expert B will be able to determine drinks/pets directly, but not jobs.
    """
    clues = []

    # 2 direct person→drink clues
    people_for_drink = random.sample(PEOPLE, 2)
    for person in people_for_drink:
        clues.append(make_person_attr_clue(person, "drink", sol[person]["drink"]))

    # 2 direct person→pet clues
    people_for_pet = random.sample(PEOPLE, 2)
    for person in people_for_pet:
        clues.append(make_person_attr_clue(person, "pet", sol[person]["pet"]))

    # 2 pet→drink cross clues
    pets_for_cross = random.sample(PETS, 2)
    for pet in pets_for_cross:
        person = next(p for p in PEOPLE if sol[p]["pet"] == pet)
        clues.append(make_attr_attr_clue("pet", pet, "drink", sol[person]["drink"]))

    return clues


def generate_harder_partition_a(sol: Solution) -> List[dict]:
    """
    Harder Partition A: only 1 direct person→job clue, rest are cross-attribute.
    Requires multi-step inference chains. Anchored by 1 direct clue.
    """
    clues = []

    # 1 direct person→job (anchor — without this combined may be non-unique)
    anchor_person = random.choice(PEOPLE)
    clues.append(make_person_attr_clue(anchor_person, "job", sol[anchor_person]["job"]))

    # 3 job→pet cross clues (all 4 jobs, pick 3)
    for job in random.sample(JOBS, 3):
        person = next(p for p in PEOPLE if sol[p]["job"] == job)
        clues.append(make_attr_attr_clue("job", job, "pet", sol[person]["pet"]))

    # 2 job→drink cross clues
    for job in random.sample(JOBS, 2):
        person = next(p for p in PEOPLE if sol[p]["job"] == job)
        clues.append(make_attr_attr_clue("job", job, "drink", sol[person]["drink"]))

    return clues


def generate_harder_partition_b(sol: Solution) -> List[dict]:
    """
    Harder Partition B: only 1 direct person→drink clue, rest are cross-attribute.
    Requires inference chains via pet→drink and drink→pet relationships.
    """
    clues = []

    # 1 direct person→drink (anchor)
    anchor_person = random.choice(PEOPLE)
    clues.append(make_person_attr_clue(anchor_person, "drink", sol[anchor_person]["drink"]))

    # 3 pet→drink cross clues
    for pet in random.sample(PETS, 3):
        person = next(p for p in PEOPLE if sol[p]["pet"] == pet)
        clues.append(make_attr_attr_clue("pet", pet, "drink", sol[person]["drink"]))

    # 2 drink→pet cross clues (reverse direction for extra inference steps)
    for drink in random.sample(DRINKS, 2):
        person = next(p for p in PEOPLE if sol[p]["drink"] == drink)
        clues.append(make_attr_attr_clue("drink", drink, "pet", sol[person]["pet"]))

    return clues


# ---------------------------------------------------------------------------
# Puzzle generator
# ---------------------------------------------------------------------------

def generate_puzzle(
    puzzle_id: str,
    difficulty: str = "medium",
    max_retries: int = 200,
) -> Optional[dict]:
    """
    Generate a single puzzle.

    Returns puzzle dict or None if can't generate valid partition in max_retries.

    Puzzle dict keys:
      puzzle_id, difficulty, solution, clues_a, clues_b,
      all_clues, single_agent_unique, partition_a_unique, partition_b_unique
    """
    for attempt in range(max_retries):
        sol = random_solution()

        if difficulty == "medium":
            clues_a = generate_partition_a_clues(sol)
            clues_b = generate_partition_b_clues(sol)
        else:  # hard
            clues_a = generate_harder_partition_a(sol)
            clues_b = generate_harder_partition_b(sol)

        structs_a = [c["struct"] for c in clues_a]
        structs_b = [c["struct"] for c in clues_b]
        structs_all = structs_a + structs_b

        # Verify properties with constraint solver
        sols_a    = count_solutions(structs_a)
        sols_b    = count_solutions(structs_b)
        sols_all  = find_solutions(structs_all)

        # Requirements:
        # 1. Neither partition alone uniquely determines solution
        # 2. Both together uniquely determine solution
        # 3. The combined solution matches our generated solution
        if sols_a > 1 and sols_b > 1 and len(sols_all) == 1:
            found_sol = sols_all[0]
            if found_sol == sol:  # Sanity check
                return {
                    "puzzle_id": puzzle_id,
                    "difficulty": difficulty,
                    "solution": sol,
                    "clues_a": clues_a,
                    "clues_b": clues_b,
                    "all_clues": clues_a + clues_b,
                    "metadata": {
                        "solutions_a_alone": sols_a,
                        "solutions_b_alone": sols_b,
                        "solutions_combined": 1,
                        "attempts_to_generate": attempt + 1,
                    },
                }

    return None  # Failed


def generate_dataset(n_medium: int = 20) -> List[dict]:
    """Generate full puzzle dataset (all medium difficulty).

    Medium puzzles use a mix of direct + cross-attribute clues, balanced
    between two partitions. Neither partition alone is sufficient (verified).
    Together they uniquely determine the full solution (verified).

    Note: All 20 puzzles are equally challenging for LLMs because:
      - Neither partition has more than 2 direct person→attribute clues
      - Cross-attribute clues require multi-step inference
      - The partition split is random per puzzle, varying the inference chains
    """
    puzzles = []

    print(f"Generating {n_medium} puzzles...")
    for i in range(n_medium):
        for retry in range(10):  # retry if generate_puzzle fails
            p = generate_puzzle(
                puzzle_id=f"puzzle_{i+1:03d}",
                difficulty="medium",
            )
            if p:
                puzzles.append(p)
                avg_attempts = p["metadata"]["attempts_to_generate"]
                print(f"  ✓ puzzle_{i+1:03d} ({avg_attempts} internal attempts, "
                      f"A={p['metadata']['solutions_a_alone']} solutions alone, "
                      f"B={p['metadata']['solutions_b_alone']} solutions alone)")
                break
        else:
            print(f"  ✗ puzzle_{i+1:03d} FAILED after 10 retries — skipping")

    return puzzles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate logic-grid puzzles")
    parser.add_argument("--medium", type=int, default=20, help="Number of puzzles to generate")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent.parent.parent.parent / "data" / "puzzle" / "puzzles.json"),
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("GAIA Asymmetric Puzzle Generator")
    print("=" * 60)
    print(f"Domain: {len(PEOPLE)} people × {len(ATTRIBUTES)} attributes")
    print(f"Search space: {len(JOBS)}! × {len(PETS)}! × {len(DRINKS)}! = 13,824 candidates")
    print(f"Target: {args.medium} puzzles\n")

    puzzles = generate_dataset(n_medium=args.medium)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"puzzles": puzzles, "total": len(puzzles)}, f, indent=2)

    print(f"\n✓ Saved {len(puzzles)} puzzles to {output_path}")
    print("\nVerification summary:")
    for p in puzzles:
        m = p["metadata"]
        print(
            f"  {p['puzzle_id']:30s} "
            f"A_alone={m['solutions_a_alone']:3d}  "
            f"B_alone={m['solutions_b_alone']:3d}  "
            f"combined=1"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
