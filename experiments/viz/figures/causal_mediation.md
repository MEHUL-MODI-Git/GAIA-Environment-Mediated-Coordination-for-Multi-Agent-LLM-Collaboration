# C2-2 — Controlled causal effect of conflict-as-task (gated by dissent)

**Method (honest):** classical NIE/NDE mediation is *not identified* here — the mediator (CONFLICT firing) is structurally entangled with the treatment (it cannot fire without the mechanism). We instead report the controlled do-style effect this designed experiment *does* identify, plus the dose-sweep gating variable.

## Controlled effect (same substrate & stressor; only conflict-as-task toggled)
- majority vote (no buffer, no conflict)        E[Y] = 0.154
- do(mechanism = OFF)  ≡ plain-blackboard       E[Y] = 0.846
- do(mechanism = ON)   ≡ GAIA                   E[Y] = 1.000
- **Controlled effect of conflict-as-task = +0.154** (+15.4 pp over an identical shared buffer)
- P(CONFLICT fires | GAIA) = 0.92  → the effect is delivered through the typed-CONFLICT→reconciler path on ~92% of episodes (the rest are trivially unanimous).

## Gating variable (dose sweep): the effect requires a dissenter
| #misled | #clean | GAIA acc | CONFLICT rate |
|---|---|---|---|
| 0 | 3 | 100% | 0% |
| 1 | 2 | 100% | 100% |
| 2 | 1 | 100% | 100% |
| 3 | 0 | 15% | 8% |

- dissenter present (≥1 clean): GAIA acc ∈ {100%, 100%, 100%} → effect ≈ +85 pp vs majority
- NO dissenter (all misled):    GAIA acc = 15% ≈ majority (15%) → controlled effect ≈ 0

**Causal characterization.** The conflict-as-task mechanism has a large, positive *controlled* effect (+15.4 pp) over an otherwise identical shared-buffer system, and that effect is **gated by the existence of a minority dissenter**: it is realized only when ≥1 agent disagrees (so a CONFLICT can be raised and reconciled) and vanishes when every agent shares the bias. This is a precise, honest causal statement backed by a controlled toggle + a dose sweep — stronger and more defensible than an (unidentified) mediation split.