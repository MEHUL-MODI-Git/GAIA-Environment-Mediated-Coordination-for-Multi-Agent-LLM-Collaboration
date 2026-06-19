# URECA Research Poster Content
## GAIA: A Multi-Agent Blackboard Architecture for Collaborative AI Problem Solving

---

## HEADER (Top of poster)

**Title:**
GAIA: Emergent Collaboration Through a Shared Blackboard in Multi-Agent AI Systems

**Author:** [YOUR NAME]
**Project ID:** [YOUR PROJECT ID]
**Supervisor:** [SUPERVISOR NAME, TITLE]
**School of Computer Science and Engineering, Nanyang Technological University**

---

## SECTION 1 — MOTIVATION / INTRODUCTION

### Problem
Large language models (LLMs) are powerful individual reasoners, but they fail silently — producing wrong answers with high confidence and no self-correction. In complex tasks like logic puzzles, coding, or multi-step math, a single agent's error is unrecoverable.

**Key questions:**
- Can multiple AI agents *collaborate* to reduce individual errors?
- Does mere *parallelism* (more agents) help, or does **structured coordination** matter?
- How can we make agent collaboration *transparent* and *auditable*?

### Existing Approaches and Their Gaps
- **Single-agent prompting** (Chain-of-Thought, ReAct): No error recovery mechanism
- **Multi-agent frameworks** (AgentVerse, AutoGen): Report accuracy gains but lack transparency into *why* collaboration helps
- **Majority voting**: Improves robustness but offers no diagnosis — fails when agents make systematic shared errors

### Our Approach: GAIA
We propose GAIA — a **shared Blackboard** coordination layer where agents post typed artifacts (plans, conflicts, reviews, evidence), enabling:
1. **Parallel independent solving** (no anchoring)
2. **Conflict detection** (aggregator reads all plans)
3. **Targeted reconciliation** (stronger model audits only disputed problems)
4. **Verifiable audit trail** (full trace of every agent's reasoning)

---

## SECTION 2 — METHODOLOGY

### System Architecture (→ Figure 1)

GAIA implements a **4-phase blackboard loop**:

**Phase 1 — Parallel Solving:**
Three Solver agents work *independently* (temperatures 0.0 / 0.3 / 0.6 for diversity). Each posts a typed `PLAN` artifact to the shared blackboard containing its reasoning chain and final answer.

**Phase 2 — Aggregation:**
The Aggregator reads all three PLAN artifacts. If all agree → posts `REVIEW` (unanimous). If any disagree → posts a `CONFLICT` signal to the blackboard.

**Phase 3 — Reconciliation (conditional):**
Only triggered on CONFLICT. The Reconciler (a more capable model, GPT-4.1) audits all three reasoning chains, identifies the erroneous step, and posts an authoritative `REVIEW` artifact.

**Phase 4 — Verification:**
The Verifier compares the REVIEW answer to ground truth and posts `EVIDENCE` (passed = True/False).

### Key Design Choices
- **Isolation in Phase 1:** Solvers do not see each other's work → prevents anchoring bias
- **Typed artifacts:** Every interaction is a named, versioned blackboard entry → full reproducibility
- **Conditional reconciliation:** Strong model only activates when needed → cost-efficient
- **Temperature diversity:** [0.0, 0.3, 0.6] increases probability of at least one solver finding the correct path

### Experimental Setup

| Task Type | Problems | Model (Solvers) | Baseline |
|-----------|----------|----------------|---------|
| Logic Puzzles (information asymmetry) | 20 | GPT-4.1-mini | Oracle / Isolated |
| GSM8K Hard Math | 20 | GPT-4.1-nano | Single agent |
| HumanEval (code generation) | 164 | GPT-4.1-mini | Single agent |
| MiniWoB++ (web automation) | 20 | GPT-4.1-mini | — |

**Ablation conditions:** Single agent | Isolated (agents, no coordination) | Majority Vote | GAIA (full system)

---

## SECTION 3 — KEY FINDINGS

### Finding 1: GAIA Consistently Outperforms Single Agent (→ Figure 2)

| Task | Single Agent | GAIA | Gain |
|------|-------------|------|------|
| Logic Puzzle | 10% | **95%** | +85pp |
| GSM8K Math | 95% | **100%** | +5pp |
| HumanEval | 58.5% | **96.4%** | +37.9pp |
| MiniWoB++ | — | **80%** | — |

*HumanEval: GAIA 96.4% = cumulative with retries on failures (pass@3 style, 3 iterations per failed problem). Single = 58.5% (first pass, no retry = effective single-attempt baseline).*

GAIA improves accuracy across **4 fundamentally different task types** — logic, math, coding, and web interaction.

---

### Finding 2: Coordination Matters, Not Agent Count (→ Figure 4)

The puzzle experiment ablation isolates the source of improvement:

| Condition | Agents | Info | Coordination | Accuracy |
|-----------|--------|------|-------------|---------|
| Oracle | 1 | Full (12 clues) | — | 100% |
| Single Partial | 1 | Partial (6 clues) | — | **10%** |
| Isolated | 8 | Partial (6 clues each) | **None** | **10%** |
| GAIA | 8 | Partial (6 clues each) | **Blackboard** | **95%** |

**Key insight:** Increasing from 1 to 8 agents without coordination yields *zero improvement* (10% → 10%). The entire 85-percentage-point gain comes from the blackboard sharing mechanism alone.

---

### Finding 3: Conflict Resolution Catches Errors Majority Vote Cannot (→ Figure 3)

On the GSM8K snail-climbing problem:
- **Solver 3 (temp=0.6)** computed day 13 — missed the extra nighttime slide on rainy days
- **Solvers 1 & 2** correctly computed day 11
- **Aggregator** detected the 2-vs-1 disagreement and posted CONFLICT
- **Reconciler (GPT-4.1)** audited all three chains, identified Solver-3's specific error (day 9 slide calculation), and posted authoritative answer 11
- **Verifier** confirmed PASS

The single agent on this problem proposed **2** (catastrophic parsing failure on its own output) → FAIL.

**Majority vote** would also pass this problem (2/3 correct), but:
- Provides **no conflict trace** — no record of which agent failed or why
- **Cannot recover** in the worst case: when 2/3 agents make the *same systematic error*, majority vote fails but GAIA's reconciler can still override with the stronger model

---

### Finding 4: Silent Failures in Single Agents

On GSM8K hard_003, the single agent reasoned correctly but extracted the wrong integer from its own output (proposed **2**, ground truth **44**). Multi-agent redundancy makes this class of failure detectable: if 3 independent agents agree on 44, the extraction error is caught.

---

## SECTION 4 — CONCLUSIONS

### What We Showed
1. **Multi-agent coordination consistently improves accuracy** across logic, math, code, and web tasks (+5pp to +85pp)
2. **Agent count alone is not sufficient** — structured coordination (blackboard) is the actual driver of improvement
3. **Conflict detection + reconciliation** provides both higher accuracy and mechanistic transparency (which agent failed, and why)
4. **Cost is efficient:** GAIA costs 3.6× a single agent but the reconciler (expensive model) only activates on conflict problems (~5% of runs in this study)

### GAIA's Differentiator vs. Prior Work
Unlike AgentVerse or AutoGen which report accuracy gains, GAIA produces a **full audit trail**:
- Which solver produced each answer
- Whether a conflict was detected
- Exactly which reasoning step the reconciler identified as erroneous

This transparency is essential for high-stakes AI deployment.

### Future Work
- Scale to AIME/AMC competition problems for deeper math stress-testing
- Apply to scientific hypothesis generation with domain-expert agents
- Extend blackboard to support long-horizon tasks (multi-turn reasoning, planning)

---

## POSTER LAYOUT SUGGESTION

**Portrait layout, 3-column:**

```
┌─────────────────────────────────────────────────────────┐
│                    TITLE + HEADER                       │
├────────────────┬───────────────────┬────────────────────┤
│  MOTIVATION    │   METHODOLOGY     │    FINDINGS        │
│  (Section 1)   │   (Section 2)     │    (Section 3)     │
│                │                   │                    │
│  - Problem     │  Fig 1:           │  Fig 2:            │
│  - Gaps        │  Architecture     │  Bar chart         │
│  - GAIA intro  │  diagram          │  (all 4 tasks)     │
│                │                   │                    │
│                │  • 4-phase loop   │  Fig 4:            │
│                │  • Isolation      │  Puzzle ablation   │
│                │  • Typed arts.    │                    │
│                │  • Exp. setup     │  Fig 3:            │
│                │                   │  Conflict trace    │
├────────────────┴───────────────────┴────────────────────┤
│                    CONCLUSIONS                          │
└─────────────────────────────────────────────────────────┘
```

---

## FIGURES TO INSERT (all in experiments/viz/poster_figures/)

1. **fig1_architecture.png** — GAIA system architecture (place in Methodology)
2. **fig2_results.png** — Results bar chart across 4 tasks (place in Findings, top)
3. **fig3_conflict_trace.png** — Conflict resolution step-by-step (place in Findings, bottom)
4. **fig4_puzzle_ablation.png** — Ablation: coordination vs. agent count (place in Findings, middle)

---

## SHORT ABSTRACT (for poster header, ~80 words)

We present GAIA, a multi-agent system in which independent AI agents collaborate through a **shared Blackboard** — a structured coordination layer where agents post typed reasoning artifacts, flag disagreements, and request reconciliation from a more capable model. Evaluated across four task types (logic puzzles, math word problems, code generation, web automation), GAIA achieves consistent accuracy gains over single-agent baselines (+5 to +85 percentage points). An ablation study shows that agent count alone does not drive improvement; the blackboard coordination mechanism is the sole source of gains.
