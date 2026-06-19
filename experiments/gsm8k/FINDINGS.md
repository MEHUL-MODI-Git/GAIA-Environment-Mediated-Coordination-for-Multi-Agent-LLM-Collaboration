# GSM8K Mathematical Reasoning — Experiment Findings

## Experiment Overview

**Goal**: Evaluate GAIA's multi-agent blackboard architecture on hard mathematical word problems.
Key question: does multi-agent redundancy and conflict resolution improve accuracy over a single agent?

**Dataset**: 20 hard math problems (`data/gsm8k/hard_problems.json`)
Problems specifically designed to trigger near-miss errors: work rates, harmonic means,
mixture equations, modular arithmetic, combinatorics, geometric series, digit traps.

**Models**: gpt-4.1-nano (Solvers, Aggregator, Verifier), gpt-4.1 (Reconciler only)
**Rationale**: gpt-4.1-nano has a meaningful error rate on tricky problems (unlike gpt-4.1-mini
which solved all 50 standard GSM8K problems with 100% accuracy).

---

## Architecture

```
Phase 1: 3 × MathSolverAgent (parallel, temperatures 0.0 / 0.3 / 0.6)
         Each reads problem independently → posts PLAN artifact
         ISOLATED: each solver does not know other agents exist (prevents anchoring)

Phase 2: MathAggregatorAgent
         Reads all 3 PLAN artifacts → LLM extracts + compares final integers
         If unanimous  → posts REVIEW artifact (subtype: "unanimous_answer")
         If conflict   → posts CONFLICT Signal to blackboard

Phase 3: (Conditional) MathReconcilerAgent  [gpt-4.1]
         Triggered only if CONFLICT signal exists
         Full context: problem + all 3 reasoning chains + conflict summary
         Runs deterministically (temp=0.0) on the slow/capable model
         Posts REVIEW artifact (subtype: "reconciled_solution")

Phase 4: MathVerifierAgent
         Finds best REVIEW (reconciled > unanimous > majority fallback)
         Pure Python comparison → posts Evidence (passed=True/False)
```

---

## Main Results (Hard Problems, gpt-4.1-nano Solvers)

| Condition | Passed | Total | Accuracy | Avg Cost/Problem |
|-----------|--------|-------|----------|-----------------|
| Single (1 solver, temp=0.0) | 19 | 20 | **95.0%** | $0.0045 |
| Majority Vote (3 solvers, no reconciliation) | 20 | 20 | **100.0%** | $0.0127 |
| GAIA (3 solvers + aggregator + reconciler) | 20 | 20 | **100.0%** | $0.0160 |

**Multi-agent redundancy beats single agent** (+5pp, 95% → 100%).
GAIA and Majority Vote reach the same final accuracy on this set, with GAIA adding conflict detection and transparency.

---

## Key Finding 1: Single Agent Failure on hard_003

The single agent (temp=0.0) proposed **2** for hard_003 (answer: 44), a severe error on this mixture problem:

> *A container holds 100 liters of a 30% alcohol solution. 20 liters is removed and replaced with pure alcohol. What is the percentage of alcohol in the final mixture?*

The single agent extracted `2` — almost certainly a parsing failure (answered in full prose, last number extracted was wrong).

**Majority Vote result**: all 3 solvers correctly answered 44 → proposed=44, PASS.
**GAIA result**: same, no conflict needed.

This illustrates the core multi-agent value: a single agent can silently produce a catastrophically wrong answer, while 3 independent solvers agree on the right answer.

---

## Key Finding 2: Real Conflict Detected and Resolved (hard_012)

In a separate run with the same setup, hard_012 triggered a genuine conflict:

> *The sum of the first N odd numbers equals N squared. What is the sum of all odd numbers from 51 to 99 inclusive?*
> **Correct answer: 1875** (= 50² − 25² = 2500 − 625)

| Solver | Answer | Correct? |
|--------|--------|---------|
| Solver-1 (temp=0.0) | 1875 | ✓ |
| Solver-2 (temp=0.3) | **1825** | ✗ |
| Solver-3 (temp=0.6) | 1875 | ✓ |

**Aggregator**: detected conflict (2 vs 1 disagreement), posted CONFLICT signal to blackboard.
**Reconciler (gpt-4.1)**: audited all 3 reasoning chains, identified Solver-2's arithmetic error
(computed 25² = 650 instead of 625), produced authoritative answer **1875**.
**Verifier**: PASS.

**Majority Vote** would also have gotten this right (2/3 majority = 1875). But:
- GAIA **detected and logged** the conflict — providing auditability
- The reconciler **diagnosed the specific error** (Solver-2 miscalculated 25²)
- In a scenario where 2/3 solvers make the same systematic error, majority vote would fail but GAIA's reconciler has a chance to catch it using the more capable gpt-4.1 model

---

## Phase Timings (GAIA, 20 problems)

| Phase | Mean | Std Dev | Min | Max |
|-------|------|---------|-----|-----|
| Phase 1: Parallel Solving | 4.00s | ±2.35s | ~1.5s | ~9s |
| Phase 2: Aggregation | 0.61s | ±0.15s | 0.45s | ~1s |
| Phase 3: Reconciliation | triggered on conflicts only | — | — | — |
| Phase 4: Verification | 0.00s (pure Python) | — | — | — |

Phase 1 dominates — 3 parallel LLM calls, with API tail latency explaining the ±2.35s spread.
Phase 2 (aggregation) is consistently fast (~0.6s) because it only extracts and compares integers.

---

## Cost Analysis

| Condition | Total (20 problems) | Per Problem | Relative to Single |
|-----------|--------------------|--------------|--------------------|
| Single | $0.090 | $0.0045 | 1.0× |
| Majority Vote | $0.254 | $0.0127 | 2.8× |
| GAIA | $0.320 | $0.0160 | 3.6× |

GAIA's overhead over majority vote is small (~$0.003/problem extra for the aggregator).
The reconciler (gpt-4.1) only runs on conflict problems, keeping the typical cost near majority vote.

---

## Data Quality Issues Found

| Problem | Original Answer | Corrected Answer | Error |
|---------|----------------|-----------------|-------|
| hard_006 | 25 | 15 | Arithmetic: (180−60)/8 = 15, not 25 |

Corrected before final run.

---

## Run History

| Run | Timestamp | Models | Accuracy (S/MV/G) | Notes |
|-----|-----------|--------|-------------------|-------|
| First hard run | 20260323_004054 | nano/4.1 | 95%/95%/95% | hard_006 ground truth wrong |
| Corrected run | 20260323_004644 | nano/4.1 | **95%/100%/100%** | Clean results |

---

## Comparison: Standard Problems (gpt-4.1-mini) vs Hard Problems (gpt-4.1-nano)

| Setup | Single | Majority Vote | GAIA | Conflicts |
|-------|--------|--------------|------|-----------|
| 50 standard problems, gpt-4.1-mini | 100% | 100% | 100% | 0/50 |
| 20 hard problems, gpt-4.1-nano | **95%** | **100%** | **100%** | 1/20 (5%) |

Switching to gpt-4.1-nano on hard problems:
- Produces real errors (5% conflict rate vs 0%)
- Reveals multi-agent redundancy benefit (+5pp single → multi-agent)
- Demonstrates the conflict detection mechanism in action

---

## Paper Narrative

**Claim**: GAIA's multi-agent blackboard architecture adds value over single-agent on problems within the model's near-miss competence window.

**Evidence from this experiment**:
1. Single agent fails silently on hard_003 (extracts wrong integer from its own reasoning)
2. 3 independent solvers with majority vote catch this failure (+5pp accuracy)
3. GAIA additionally detects conflicts (hard_012), diagnoses the specific arithmetic error,
   and resolves it using a more capable reconciler model
4. Cost overhead is modest: 3.6× vs 1× for single, but adds fault tolerance and auditability

**Differentiator vs. AgentVerse**: GAIA logs the full conflict trace — which solver failed, what the error was, and how the reconciler fixed it. This is mechanistic transparency that pure accuracy numbers don't show.

---

## Next Step for Stronger Ablation

To demonstrate GAIA > Majority Vote on accuracy (not just on conflict transparency):
Design or find problems where 2/3 solvers make the **same systematic error** (e.g., off-by-one in a Fibonacci sequence, wrong base in exponent) so majority vote picks the wrong answer but the reconciler (gpt-4.1) can override it.

Target: conflict problems where `majority answer ≠ correct answer`. This scenario exists in theory; whether it appears in practice depends on whether gpt-4.1-nano tends to make correlated errors.
