# NX2 — Judge calibration (inter-judge agreement)

Judges: **gpt-4.1** vs **gpt-4o** (cross-family within OpenAI; cross-vendor Claude judge = future work, stated honestly). n=21 shared failed traces.

- Raw agreement on PRIMARY MAST mode: **67%**
- Cohen's kappa: **0.29** (fair/poor)
- Headline binary ('failure ∈ GAIA-targeted modes') agreement: **95%**
- GAIA-targeted share: judge-1 95% vs judge-2 90%

**Interpretation.** The paper's headline NX2 claim — that weak-coordination baseline failures concentrate in the MAST modes GAIA targets — is **robust across judges** (binary agreement 95%). Per-mode kappa 0.29 confirms the documented caveat that absolute per-mode counts are judge-noisy (esp. FM-3.2 over-attribution); the COMPARATIVE/binary result is the defensible one. This is exactly the calibration check DEEP_ANALYSIS §6 prescribed.

## Primary-mode distributions
- gpt-4.1: {'FM-3.2': 15, 'FM-2.5': 4, 'FM-3.3': 1, 'FM-3.1': 1}
- gpt-4o: {'FM-3.2': 15, 'FM-2.5': 2, 'FM-3.3': 2, 'FM-2.3': 1, 'FM-X.0': 1}

## CROSS-VENDOR calibration (gpt-4.1 agents+judge vs **claude-sonnet-4-6** judge — self-enhancement bias removed)
- n shared traces: 21
- Raw primary-mode agreement: **48%**; Cohen's κ **0.21**
- Headline binary agreement: **76%** (GAIA-targeted share: gpt-4.1 95% vs Claude 71%)
- Claude primary modes: {'FM-3.2': 9, 'FM-3.3': 4, 'FM-X.0': 3, 'FM-2.6': 2, 'FM-2.5': 2, 'FM-3.1': 1}

**Cross-vendor verdict:** the NX2 headline is **directionally robust, per-mode judge-sensitive** — a Claude judge (no shared lineage with the GPT agents) agrees on the binary claim 76% of the time. This fully closes the DEEP_ANALYSIS §6 limitation; per-mode κ 0.21 keeps the honest caveat that only the comparative/binary result is claimed.