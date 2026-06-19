# C4-2 — Behavioural-taxonomy coding (2 cross-vendor judges, n=70)

## Inter-judge reliability (Cohen's κ per code; gpt-4.1 vs claude-sonnet-4-6)
| code | κ |
|---|---|
| IND | 0.327 |
| ANC | 1.0 |
| HDG | 1.0 |
| CAP | 1.0 |
| SCR | 0.489 |
| FCF | 0.0 |
| TPF | 0.552 |
| TPD | 0.282 |

## Behaviour frequency by group (high-precision consensus = both judges agree)

| group | n | IND | ANC | HDG | CAP | SCR | FCF | TPF | TPD |
|---|---|---|---|---|---|---|---|---|---|
| misled-solver | 33 | 0.12 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.36 | 0.03 |
| clean-solver | 16 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| reconciler | 9 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 |
| aggregator | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Reading
- Expected & to-verify: misled-solver high TPF/FCF (follows trap, confidently wrong), clean-solver high IND/TPD, reconciler high SCR/TPD/IND (audits & re-derives). Capitulation (CAP) in solvers = the conformity failure GAIA's structure must resist.
- κ: codes with κ≥0.6 are reliably coded; low-κ codes (e.g. FCF, inherently subjective) are reported but flagged as judge-sensitive — honest, per qualitative-coding standards. Consensus = both-judge intersection (high precision, conservative).
- This is a *behavioural repertoire* per role — an inferred agent-behaviour structure not targeted by any prior experiment.