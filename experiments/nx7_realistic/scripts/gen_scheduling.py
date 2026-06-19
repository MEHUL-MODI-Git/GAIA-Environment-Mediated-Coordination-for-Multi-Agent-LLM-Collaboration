#!/usr/bin/env python3
"""NX7 data — realistic multi-constraint meeting-scheduling problems.

A non-toy *agentic planning* task (the kind in TAU-bench / NaturalPlan-style
benchmarks): given N people with busy intervals, working hours, a required
duration, and side constraints, find a start time that satisfies ALL
constraints. Ground truth is a programmatic checker (like the puzzle
verifier) — there may be several valid slots; a proposal is correct iff it
satisfies every constraint. Each generated instance is guaranteed to have
≥1 valid slot (constructed by planting one, then adding distractor busy
blocks). 30 instances, 3 difficulty tiers.
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
OUT = ROOT/"data"/"scheduling"/"problems.json"
DAY_START, DAY_END = 9 * 60, 18 * 60       # 09:00–18:00 in minutes


def hhmm(m): return f"{m//60:02d}:{m%60:02d}"


def gen(seed, n_people, n_busy, duration):
    rng = random.Random(seed)
    # plant a guaranteed common free slot
    slot = rng.randrange(DAY_START, DAY_END - duration, 30)
    people = []
    for p in range(n_people):
        busy = []
        # add busy blocks that AVOID the planted slot
        for _ in range(n_busy):
            for _try in range(40):
                s = rng.randrange(DAY_START, DAY_END - 30, 30)
                d = rng.choice([30, 60, 90])
                e = min(s + d, DAY_END)
                if e <= slot or s >= slot + duration:    # disjoint from slot
                    busy.append([s, e]); break
        busy.sort()
        people.append({"name": f"P{p+1}", "busy": [[hhmm(a), hhmm(b)] for a, b in busy]})
    return {
        "problem_id": f"sched_{seed:03d}",
        "n_people": n_people,
        "duration_min": duration,
        "working_hours": [hhmm(DAY_START), hhmm(DAY_END)],
        "people": people,
        "instruction": (
            f"Find a meeting start time so all {n_people} people are free for "
            f"a {duration}-minute meeting, within working hours "
            f"{hhmm(DAY_START)}–{hhmm(DAY_END)}. Output the start time as "
            f"HH:MM on the last line as: FINAL: HH:MM"),
        "_planted_slot": hhmm(slot),       # not shown to solvers; for sanity
    }


def main():
    probs = []
    seed = 1
    tiers = [(3, 3, 30, 10), (4, 4, 60, 10), (5, 5, 60, 10)]  # (ppl,busy,dur,count)
    for n_people, n_busy, dur, count in tiers:
        for _ in range(count):
            probs.append(gen(seed, n_people, n_busy, dur)); seed += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(probs, open(OUT, "w"), indent=2)
    print(f"Wrote {len(probs)} scheduling problems -> {OUT}")
    print("tiers: 3p/30m, 4p/60m, 5p/60m  (x10 each)")


if __name__ == "__main__":
    main()
