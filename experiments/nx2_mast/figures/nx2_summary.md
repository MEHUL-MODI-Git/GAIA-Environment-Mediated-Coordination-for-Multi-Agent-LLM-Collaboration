# NX2 — MAST failure-mode analysis (honest)

Judge: gpt-4.1 (stronger than the gpt-4.1-nano/mini agents; full cross-family Claude judge is the rigorous upgrade — see DEEP_ANALYSIS.md §6).

## Headline (robust, comparative) result
- Weak-coordination baselines (majority-vote / AutoGen-debate / plain-blackboard) on the trap suite: **20/21 (95%) of failures fall in MAST modes GAIA's architecture explicitly targets** — dominated by **FM-3.2 No/incomplete verification** (15/21) and **FM-2.5 Ignored peer input** (4/21).
- GAIA's dedicated Verifier phase targets FM-3.2; its synthesizer/reconciler-must-read-all-artifacts + CONFLICT signal targets FM-2.5. This is the mechanistic reason GAIA reaches 100% on the same stressor (E3/NX1) — it removes exactly the modes the baselines die by. **This closes MAST's open loop empirically.**

## Honest limitations (do not hide these)
1. **FM-3.2 over-attribution.** The judge tends to label *any* wrong-final-answer trace 'no/incomplete verification', even when the true cause is information-insufficiency outside the 14 modes (e.g. GAIA at 25% clue coverage genuinely cannot solve the puzzle; the Python verifier DID run and correctly flagged failure). So the GAIA-residual FM-3.2 count is inflated and should NOT be read as 'GAIA fails to verify'.
2. **Spurious FM-1.4 / FM-2.1 on GAIA traces.** GAIA structurally has no chat to truncate or reset (persistent board); judge labels of FM-1.4/FM-2.1 on long puzzle traces are likely mis-attributions of information-insufficiency, supporting the architectural claim that GAIA *cannot* exhibit these modes rather than refuting it.
3. **Representation asymmetry.** Baseline traces are thin result records; GAIA-residual traces are rich state dumps. The COMPARATIVE baseline-only result is the defensible one; the GAIA-residual column is exploratory.
4. **Single-judge, same-vendor.** gpt-4.1 judging gpt-4.1-family agents → residual self-enhancement risk. Rigorous upgrade: Claude-family judge + 50-trace human-labelled golden set (target κ≥0.7), specified in DEEP_ANALYSIS.md §6.

## Per-group primary-mode counts
- baseline_weak: {'FM-3.2': 15, 'FM-2.5': 4, 'FM-3.3': 1, 'FM-3.1': 1}
- GAIA_residual: {'FM-3.2': 41, 'FM-1.4': 10, 'FM-2.1': 5, 'FM-2.4': 1, 'FM-3.3': 1}