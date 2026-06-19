# NX1 — Mechanism attribution (identical 2-misled+1-clean stressor)

| Condition | Accuracy [95% CI] | Cost |
|---|---|---|
| Single (1 clean, no stressor) | 100.0% [100%, 100%] | $0.042 |
| Majority vote (2 misled+1 clean) | 15.4% [0%, 38%] | $0.106 |
| Debate (AutoGen-style, 2 rounds) | 38.5% [15%, 62%] | $0.384 |
| Plain blackboard (2025 BB-paper design) | 84.6% [62%, 100%] | $0.145 |
| GAIA (+ conflict-as-task) | 100.0% [100%, 100%] | $0.215 |

**Reading:** debate (38.5%) barely beats majority (15.4%) — iterative revision *entrenches* the shared error via conformity. A plain shared buffer recovers most cases (84.6%) because the synthesizer can see the lone correct chain. GAIA's structured conflict-as-task closes the final gap to 100%. The active ingredient is **not** 'a blackboard' nor 'debate' — it is the typed CONFLICT signal escalating to reconciliation.