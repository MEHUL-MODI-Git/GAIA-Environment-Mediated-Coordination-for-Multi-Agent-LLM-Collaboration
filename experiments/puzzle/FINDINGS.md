# Asymmetric Information Puzzle — Experiment Findings

Benchmark: 20 logic-grid puzzles (4 people × 3 attributes), 3 conditions, 60 total runs.
Models: `gpt-4.1-mini` (Experts, Critic, Verifier), `gpt-4.1` (Synthesizers).
Date: 2026-03-22.

---

## Results Summary

| Condition | Accuracy | N | Avg Cost | Description |
|-----------|----------|---|----------|-------------|
| **Single** | 100% | 20 | $0.006 | 1 agent, all 12 clues, synthesizer prompt |
| **Isolated** | 10% | 20 | ~$0.020 | 2 agents, each sees only 6 clues, no sharing |
| **GAIA** | 95% | 20 | $0.059 | 8 agents, shared blackboard |

**Key finding**: Information asymmetry alone causes performance to collapse (10%). GAIA's shared blackboard recovers 95% of the theoretical upper bound (100%) despite each agent having only half the clues.

---

## Emergent Behaviors

### 1. Confident Wrong Consensus (puzzle_002 — the only GAIA failure)

**What happened**: Both synthesizers produced identical wrong answers. The Critic verified they agreed and posted AGREE. The whole pipeline was confidently and unanimously wrong.

**Root cause**: puzzle_002's drink clues were split asymmetrically. Partition A had no direct person→drink links (only "doctor drinks juice, teacher drinks coffee"). Partition B had Alice→water, Carol→tea. Expert A agents correctly reported they couldn't deduce Alice/Carol's drinks from their partition. But both synthesizers failed to pull those values from Partition B's expert deductions — outputting `drink=unknown` for Alice and Carol.

**The blackboard had all the information**. The failure was in synthesis, not information availability. Both synthesizers made the same reasoning error independently, so the Critic found consensus on a wrong answer.

**Paper significance**: Demonstrates a failure mode of multi-agent consensus systems — parallel agents can reinforce each other's mistakes. Unanimous agreement is not a correctness signal.

```
Proposed (both synthesizers):
  Alice:  job=artist,   pet=dog,  drink=unknown  ← WRONG (correct: water)
  Carol:  job=engineer, pet=cat,  drink=unknown  ← WRONG (correct: tea)
  Bob:    job=teacher,  pet=bird, drink=coffee   ✓
  Dave:   job=doctor,   pet=fish, drink=juice    ✓

Ground truth drinks: Alice=water, Carol=tea (both in Partition B clues)
```

---

### 2. The Conflict-Resolution Path Was Never Triggered

**What happened**: The Critic returned AGREE on all 20 puzzles. Phase 3b (re-synthesis after conflict) was never executed.

**Why**: The two synthesizers (temperatures 0.1 and 0.3) consistently converged to the same answer when given the same expert deductions. Temperature diversity was insufficient to produce divergent solutions on these puzzles.

**Implication**: The AGREE/CONFLICT mechanism and re-synthesis path are architecturally sound but require harder tasks (more ambiguous deductions, more agents, larger grids) to be exercised. For the paper, this is worth noting — the conflict path may add more value on harder benchmarks.

---

### 3. True Parallelism Confirmed

All 4 Expert agents claim their tasks within **≤10ms** of each other and execute fully in parallel. The bottleneck is always the slowest expert.

Typical spread between first and last expert finishing: **0.6–2.9 seconds** (out of ~7s total Phase 1 time).

```
puzzle_004:  spread = 0.88s  (tightest)
puzzle_016:  spread = 2.87s  (widest, excluding outlier)
puzzle_020:  spread = 11.33s (API latency spike — see below)
```

---

### 4. API Latency Spike (puzzle_020)

puzzle_020 took **46.9s** vs 20.1s average. Two experts finished in ~6.7s (normal). The other two took **17.2s and 17.8s** — same prompt, same token count (~985 tokens), 2.6× slower. This is an OpenAI server latency spike, not a reasoning difference.

The system correctly waited for the slowest agent before proceeding to Phase 2 (the bottleneck behavior of `asyncio.gather`). Cost was unaffected ($0.0573 — within normal range) because cost is driven by token count, not latency.

**Implication**: Wall-clock time is highly sensitive to tail latency in parallel execution. Cost is not.

---

### 5. Phase Timing Stability

Despite varying puzzle structures, phase durations are remarkably stable (excluding puzzle_020):

| Phase | Mean | Std | Description |
|-------|------|-----|-------------|
| Phase 1 (4 experts parallel) | 7.3s | ±1.4s | All 4 experts run in parallel |
| Phase 2 (2 synthesizers parallel) | 7.1s | ±1.3s | Both synthesizers run in parallel |
| Phase 3 (critic) | 1.4s | ±0.3s | Single fast LLM call (gpt-4.1-mini) |
| Phase 4 (verifier) | 3.3s | ±0.8s | LLM sanity check + Python solver |
| **Total** | **~20s** | ±2s | End-to-end per puzzle |

Phase 1 and Phase 2 take nearly identical wall-clock time: Phase 1 runs 4 mini-model calls, Phase 2 runs 2 full-model calls — they balance out.

---

### 6. Cost Structure

- **Per puzzle (GAIA)**: $0.059 average, range $0.056–$0.064 (only 1.14× spread)
- **Despite 2.3× duration variation** (18–47s), cost is highly predictable — driven by fixed 8 LLM calls per puzzle
- **Phase cost breakdown** (approximate):
  - Phase 1 (4 × gpt-4.1-mini, ~950 tokens each): ~$0.022
  - Phase 2 (2 × gpt-4.1, ~2500 tokens each): ~$0.027
  - Phase 3 (1 × gpt-4.1-mini, ~1900 tokens): ~$0.005
  - Phase 4 (1 × gpt-4.1-mini, ~800 tokens): ~$0.004
- **Synthesis accounts for ~46% of cost** despite being only 2 of 8 LLM calls — the expensive gpt-4.1 calls

---

### 7. Isolated Condition: 2 Lucky Guesses

The 2 puzzles that passed in the isolated condition (10%) were likely cases where gpt-4.1 correctly guessed between the two equally-consistent solutions for its partition. Not evidence of reasoning — mathematically, each partition has exactly 2 valid solutions, so random selection gives 50% on the specific attributes where ambiguity exists. With 12 attributes total and only 2 attributes ambiguous per partition, the expected random pass rate is approximately 1/4 (25%) per synthesizer, but both have to be right simultaneously — observed 10% is consistent.

---

## Experiment Design Notes

- **Partition guarantee**: Every puzzle was validated: Partition A alone → 2 solutions, Partition B alone → 2 solutions, combined → exactly 1 solution. This is enforced by the brute-force constraint solver in `generate_puzzles.py`.
- **Single = oracle upper bound**: Giving all 12 clues to one agent with the synthesizer prompt format yields 100%. This establishes the ceiling and shows the task itself isn't inherently hard for LLMs — it's the information asymmetry that creates difficulty.
- **Isolated = information-asymmetry floor**: 10% with split information and no sharing.
- **GAIA = 95%**: 9.5× improvement over isolated, 5% gap from oracle.

---

## Open Questions for Paper

1. Would a harder puzzle (6 people × 4 attributes) trigger the conflict-resolution path?
2. Does the "confident wrong consensus" failure mode correlate with specific clue structures (e.g., cross-partition drink chains)?
3. Is the 5% gap between GAIA (95%) and oracle (100%) irreducible, or solvable with better synthesizer prompts?
4. The isolated condition only gets 10% — does adding more isolated agents (without sharing) improve this at all?
