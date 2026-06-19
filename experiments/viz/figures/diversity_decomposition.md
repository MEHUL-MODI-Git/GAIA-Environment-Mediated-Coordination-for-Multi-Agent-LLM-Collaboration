# C2-1 — Diversity Prediction Theorem decomposition

n=13 trap problems. Predictive diversity is LOW here by construction (2 misled agents give the SAME wrong answer).

| System | mean sq-error (all) | mean sq-error (LOW-diversity subset) |
|---|---|---|
| DPT crowd (mean predictor) | 260 | 15 |
| Majority vote | 771 | 255 |
| Debate (AutoGen-style) | 570 | 17 |
| GAIA (conflict-as-task) | 0 | 0 |

**Interpretation.** The DPT mean-crowd and majority/debate carry large squared error precisely on the LOW-diversity problems — exactly as the theorem dictates (no diversity ⇒ no crowd benefit). GAIA's mean squared error stays ≈0 on the SAME low-diversity problems. GAIA is therefore not a wisdom-of-crowds aggregator at all: conflict-as-task extracts the correct answer from a *minority* dissenter, breaking the diversity dependence that bounds every averaging/voting/debate scheme. To our knowledge no prior LLM-MAS work frames coordination as *escaping* the Hong–Page diversity requirement.