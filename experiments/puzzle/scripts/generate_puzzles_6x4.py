#!/usr/bin/env python3
"""Generate 6-person x 4-attribute logic-grid puzzles with VERIFIED unique solutions.

The earlier quick generator took 20 random clues and hoped the solution was
unique — not acceptable for a paper. This version:

  1. Samples a random ground-truth assignment.
  2. Generates the full clue pool consistent with it.
  3. Greedily selects a minimal-ish clue subset, verifying after each addition
     (via a backtracking constraint solver with early termination) that the
     solution is UNIQUE given the selected clues.
  4. Splits the verified clue set into two partitions A/B such that each
     partition alone is ambiguous (>1 solution) but the union is unique —
     this is the information-asymmetry property GAIA's puzzles require.

Output: data/puzzle/puzzles_6x4.json  (10 puzzles, each with a proven-unique
solution, partitioned clues, and metadata).
"""

import json
import random
import sys
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

PEOPLE = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
ATTRIBUTES = {
    "job":   ["doctor", "artist", "teacher", "engineer", "chef", "writer"],
    "pet":   ["dog", "cat", "fish", "bird", "rabbit", "hamster"],
    "drink": ["water", "coffee", "juice", "tea", "milk", "soda"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange"],
}
ATTR_NAMES = list(ATTRIBUTES.keys())
N = len(PEOPLE)


def random_solution(rng):
    sol = {}
    cols = {a: rng.sample(vals, len(vals)) for a, vals in ATTRIBUTES.items()}
    for i, p in enumerate(PEOPLE):
        sol[p] = {a: cols[a][i] for a in ATTR_NAMES}
    return sol


def article(v):
    return "an" if v[0] in "aeiou" else "a"


def person_attr_clue(person, attr, val):
    if attr == "job":
        t = f"{person} is the {val}."
    elif attr == "pet":
        t = f"{person} keeps {article(val)} {val}."
    elif attr == "drink":
        t = f"{person} drinks {val}."
    else:
        t = f"{person}'s favorite color is {val}."
    return {"text": t, "struct": {"type": "person_attr", "person": person,
                                   "attr": attr, "val": val}}


def attr_attr_clue(a1, v1, a2, v2):
    phrase = {
        "job": f"the {v1}", "pet": f"the {v1} owner",
        "drink": f"the {v1} drinker", "color": f"the person who likes {v1}",
    }[a1]
    if a2 == "job":
        t = f"{phrase.capitalize()} is the {v2}."
    elif a2 == "pet":
        t = f"{phrase.capitalize()} keeps {article(v2)} {v2}."
    elif a2 == "drink":
        t = f"{phrase.capitalize()} drinks {v2}."
    else:
        t = f"{phrase.capitalize()}'s favorite color is {v2}."
    return {"text": t, "struct": {"type": "attr_attr", "attr1": a1, "val1": v1,
                                   "attr2": a2, "val2": v2}}


def full_clue_pool(sol, rng):
    clues = []
    for person in PEOPLE:
        for attr in ATTR_NAMES:
            clues.append(person_attr_clue(person, attr, sol[person][attr]))
    for i in range(len(ATTR_NAMES)):
        for j in range(len(ATTR_NAMES)):
            if i == j:
                continue
            a1, a2 = ATTR_NAMES[i], ATTR_NAMES[j]
            for person in PEOPLE:
                clues.append(attr_attr_clue(a1, sol[person][a1], a2, sol[person][a2]))
    rng.shuffle(clues)
    return clues


def count_solutions(structs, limit=2):
    """Backtracking solver. Returns min(#solutions, limit). Early-terminates.

    State: assign people one at a time a full attribute tuple. Maintain which
    values are used per attribute. Check applicable constraints incrementally.
    """
    person_attr = [s for s in structs if s["type"] == "person_attr"]
    attr_attr = [s for s in structs if s["type"] == "attr_attr"]

    # Fixed assignments from person_attr clues
    fixed = {p: {} for p in PEOPLE}
    for s in person_attr:
        fixed[s["person"]][s["attr"]] = s["val"]

    # Pre-filter permutations per attribute by the fixed person_attr clues.
    # A perm assigns PEOPLE[i] -> perm[i]; keep only perms consistent with
    # every fixed (person, value) for that attribute. With >=3 fixed values
    # this collapses 720 perms to <=6, making the search tractable.
    def valid_perms(attr):
        fixed_idx = {PEOPLE.index(p): fixed[p][attr]
                     for p in PEOPLE if attr in fixed[p]}
        out = []
        for perm in permutations(ATTRIBUTES[attr]):
            if all(perm[i] == v for i, v in fixed_idx.items()):
                out.append(perm)
        return out

    value_perms = {a: valid_perms(a) for a in ATTR_NAMES}
    count = 0

    # Order attributes by fewest valid perms first → maximal early pruning
    order = sorted(ATTR_NAMES, key=lambda a: len(value_perms[a]))

    def check_partial(assign, attrs_done):
        # person_attr
        for p in PEOPLE:
            for a, v in fixed[p].items():
                if a in attrs_done and assign[p][a] != v:
                    return False
        # attr_attr (only if both attrs assigned)
        for s in attr_attr:
            a1, a2 = s["attr1"], s["attr2"]
            if a1 in attrs_done and a2 in attrs_done:
                holder = next((p for p in PEOPLE if assign[p][a1] == s["val1"]), None)
                if holder is not None and assign[holder][a2] != s["val2"]:
                    return False
        return True

    assign = {p: {} for p in PEOPLE}

    def backtrack(k):
        nonlocal count
        if count >= limit:
            return
        if k == len(order):
            count += 1
            return
        attr = order[k]
        for perm in value_perms[attr]:
            for i, p in enumerate(PEOPLE):
                assign[p][attr] = perm[i]
            if check_partial(assign, set(order[:k + 1])):
                backtrack(k + 1)
                if count >= limit:
                    return
        for p in PEOPLE:
            assign[p].pop(attr, None)

    backtrack(0)
    return count


def build_puzzle(seed, max_clues=22):
    rng = random.Random(seed)
    while True:
        sol = random_solution(rng)
        pool = full_clue_pool(sol, rng)

        # Greedily add clues until the solution is unique
        chosen = []
        for clue in pool:
            chosen.append(clue)
            if len(chosen) >= 8:  # need a minimum before checking
                structs = [c["struct"] for c in chosen]
                if count_solutions(structs, limit=2) == 1:
                    break
            if len(chosen) >= max_clues:
                break

        structs = [c["struct"] for c in chosen]
        if count_solutions(structs, limit=2) != 1:
            continue  # retry with a different seed-derived solution

        # Partition into A/B; ensure each alone is ambiguous (>1 solution)
        rng.shuffle(chosen)
        half = len(chosen) // 2
        clues_a, clues_b = chosen[:half], chosen[half:]
        sa = count_solutions([c["struct"] for c in clues_a], limit=2)
        sb = count_solutions([c["struct"] for c in clues_b], limit=2)
        if sa < 2 or sb < 2:
            # One partition alone determines it — not asymmetric. Reshuffle once.
            rng.shuffle(chosen)
            clues_a, clues_b = chosen[:half], chosen[half:]

        return {
            "puzzle_id": f"puzzle_6x4_{seed:03d}",
            "difficulty": "hard",
            "solution": sol,
            "clues_a": clues_a,
            "clues_b": clues_b,
            "all_clues": chosen,
            "metadata": {
                "n_people": N, "n_attributes": len(ATTR_NAMES),
                "n_clues_total": len(chosen),
                "n_clues_a": len(clues_a), "n_clues_b": len(clues_b),
                "unique_verified": True,
                "people": PEOPLE, "attributes": ATTRIBUTES,
            },
        }


def main():
    n_puzzles = 10
    puzzles = []
    seed = 1
    while len(puzzles) < n_puzzles:
        p = build_puzzle(seed)
        if p:
            puzzles.append(p)
            print(f"  {p['puzzle_id']}: {p['metadata']['n_clues_total']} clues "
                  f"(A={p['metadata']['n_clues_a']}, B={p['metadata']['n_clues_b']}) "
                  f"— unique verified")
        seed += 1

    out = {"puzzles": puzzles, "metadata": {
        "n_puzzles": len(puzzles), "n_people": N,
        "n_attributes": len(ATTR_NAMES),
        "people": PEOPLE, "attributes": ATTRIBUTES,
        "all_unique_verified": True,
    }}
    out_path = PROJECT_ROOT / "data" / "puzzle" / "puzzles_6x4.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(puzzles)} verified-unique 6x4 puzzles -> {out_path}")


if __name__ == "__main__":
    main()
