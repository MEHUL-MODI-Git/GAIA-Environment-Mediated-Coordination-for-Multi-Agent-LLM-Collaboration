# Reasoning-Chain Samples (for paper appendix)


## Episode: puzzle_001

### Deduction-auditor trust analysis

```
## CONTRADICTION PAIRS
- Expert-A-2: "Alice: job=doctor" CONTRADICTS FaultyExpert-B: "Alice: job=teacher"
- Expert-A-1: "Alice: job=doctor" CONTRADICTS FaultyExpert-B: "Alice: job=teacher"
- Expert-A-2: "Alice: job=doctor" CONTRADICTS Expert-B-1: "Alice: job=UNKNOWN"
- Expert-A-1: "Alice: job=doctor" CONTRADICTS Expert-B-1: "Alice: job=UNKNOWN"
- Expert-A-2: "Bob: job=artist" CONTRADICTS Expert-B-1: "Bob: job=UNKNOWN"
- Expert-A-1: "Bob: job=artist" CONTRADICTS Expert-B-1: "Bob: job=UNKNOWN"
- Expert-A-2: "Bob: job=artist" CONTRADICTS FaultyExpert-B: "Bob: job=UNKNOWN"
- Expert-A-1: "Bob: job=artist" CONTRADICTS FaultyExpert-B: "Bob: job=UNKNOWN"

## TRUST SCORES
- Expert-A-2: 1.0 — All deductions are consistent with the majority and internally consistent.
- Expert-B-1: 1.0 — All deductions are consistent with the majority and internally consistent; jobs are left unknown due to clue partition.
- FaultyExpert-B: 0.0 — Assigns Alice two different jobs (teacher and artist), directly contradicting other experts and itself.
- Expert-A-1: 1.0 — All deductions are consistent with the majority and internally consistent.

## SUSPECTED FAULTY EXPERT
Suspected faulty: FaultyExpert-B
Reason: FaultyExpert-B assigns Alice two different jobs (teacher and artist), which is impossible, and this directly contradicts the consistent assignments of the other experts.

## CONSISTENCY SUMMARY
The majority of experts (Expert-A-2, Expert-A-1, and Expert-B-1) are consistent with each other, with diffe
```


## Episode: puzzle_002

### Deduction-auditor trust analysis

```
## CONTRADICTION PAIRS
- Expert-A-1: "Alice: job=artist" CONTRADICTS FaultyExpert-B: "Alice: job=CONTRADICTION (doctor and teacher)"
- Expert-A-1: "Carol: job=engineer" CONTRADICTS FaultyExpert-B: "Carol: job=UNKNOWN"
- Expert-A-1: "Carol: pet=cat" CONTRADICTS FaultyExpert-B: "Carol: pet=cat" (NO CONTRADICTION HERE, SAME FACT)
- Expert-A-1: "Dave: pet=UNKNOWN" CONTRADICTS FaultyExpert-B: "Dave: pet=fish"
- Expert-A-1: "Bob and Dave are either doctor or teacher" CONTRADICTS FaultyExpert-B: "Alice is both doctor and teacher"
- Expert-A-2: "Alice: job=artist" CONTRADICTS FaultyExpert-B: "Alice: job=CONTRADICTION (doctor and teacher)"
- Expert-A-2: "Carol: job=engineer" CONTRADICTS FaultyExpert-B: "Carol: job=UNKNOWN"
- Expert-A-2: "Carol: pet=cat" CONTRADICTS FaultyExpert-B: "Carol: pet=cat" (NO CONTRADICTION HERE, SAME FACT)
- Expert-A-2: "Dave: pet=UNKNOWN" CONTRADICTS FaultyExpert-B: "Dave: pet=fish"
- Expert-A-2: "Bob and Dave's jobs are doctor and teacher" CONTRADICTS FaultyExpert-B: "Alice is both doctor and teacher"
- Expert-B-1: "Alice: drink=water" CONTRADICTS FaultyExpert-B: "Alice: drink=UNKNOWN"
- Expert-B-1: "Carol: drink=tea" CONTRADICTS FaultyExpert-B: "Carol: drink=tea" (NO CONTRADICTION HERE, SAME FACT)
- Expert-B-1: "Dave: drink=juice" CONTRADICTS FaultyExpert-B: "Dave: drink=juice" (NO CONTRADICTION HERE, SAME FACT)
- Expert-B-1: "Dave: pet=fish" CONTRADICTS Expert-A-1: "Dave: pet=UNKNOWN"
- Expert-B-1: "Dave: pet=fish" CONTRADICTS Expert-A-2: "Dave: pet=UNKNO
```


## Episode: puzzle_003

### Deduction-auditor trust analysis

```
## CONTRADICTION PAIRS
- FaultyExpert-B: "Alice: job=CONTRADICTION" CONTRADICTS Expert-A-1: "Alice: job=teacher" and Expert-A-2: "Alice: job=teacher"
- FaultyExpert-B: "Bob: pet=bird" and "Carol: pet=cat" CONTRADICTS Expert-A-1 and Expert-A-2: "Bob: pet=UNKNOWN", "Carol: pet=UNKNOWN" (since A-1 and A-2 do not assign these pets, but do not directly contradict; this is not a direct contradiction, just a difference in deduction progress)
- Expert-B-1: "Alice: drink=coffee" CONTRADICTS Expert-A-1 and Expert-A-2: "Alice: drink=UNKNOWN" (but not a direct contradiction, as A-1 and A-2 simply do not assign a drink)
- Expert-B-1: "Carol: pet=cat" and "Carol: drink=juice" is consistent with A-1 and A-2: "Carol: drink=juice", "Carol: pet=UNKNOWN" (no contradiction)
- FaultyExpert-B: "Alice cannot be both artist and engineer simultaneously" is not a contradiction, but a statement about a contradiction in the clues they received.

## TRUST SCORES
- FaultyExpert-B: 0.0 — Multiple deductions (especially Alice's job) directly contradict the majority consensus (A-1 and A-2), and this expert explicitly reports a contradiction in their clues.
- Expert-B-1: 0.5 — Some deductions (Alice drinks coffee) are not confirmed by the majority, but do not directly contradict; overall, deductions are plausible but less certain.
- Expert-A-1: 1.0 — All deductions are consistent with the majority (A-2), no contradictions found.
- Expert-A-2: 1.0 — All deductions are consistent with the majority (A-1), no con
```
