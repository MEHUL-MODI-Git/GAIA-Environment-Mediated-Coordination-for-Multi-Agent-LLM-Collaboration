# C4-3b — Elicited-confidence calibration (positive counterpart to C4-3's lexical negative)

Source: misled-solver elicited CONFIDENCE: X at N=6 (C4-1).  n=78 misled samples.

| metric | value |
|---|---|
| mean elicited confidence | 0.992 |
| accuracy | 0.179 |
| ECE | 0.812 |
| confidently-wrong rate (conf≥0.8 & wrong) | 0.821 |

## Reliability table
| conf bin | n | mean conf | empirical acc |
|---|---|---|---|
| [0.0,0.2) | 0 | — | — |
| [0.2,0.4) | 0 | — | — |
| [0.4,0.6) | 0 | — | — |
| [0.6,0.8) | 0 | — | — |
| [0.8,1.0) | 78 | 0.992 | 0.179 |

## Reading
- **Also non-discriminative (honest negative, consistent with C4-3).** Even explicitly elicited confidence clusters near a single value and does not separate right from wrong on these traps: the misled solver is *confidently wrong* — calibration cannot rescue a correlated bias, which is exactly why GAIA's structural conflict-as-task (not confidence weighting) is the mechanism that recovers truth. This strengthens, not weakens, the C4-1 thesis.