# NX4 + Stats — Headline results with bootstrap 95% CIs

Scope: CONSTRUCTED diagnostic suites (hand-built traps/puzzles), not random population samples. CIs = within-suite precision (10k bootstrap resamples of the per-problem pass vector).

| Exp | Condition | n | Accuracy [95% CI] | $/task | On Pareto front |
|---|---|---|---|---|---|
| E3 | single | 13 | 100.0% [100.0%, 100.0%] | $0.0032 | ★ |
| E3 | gaia | 13 | 100.0% [100.0%, 100.0%] | $0.0165 |  |
| E3 | majority_vote | 13 | 15.4% [0.0%, 38.5%] | $0.0081 |  |
| E4 | gaia_c100 | 20 | 90.0% [75.0%, 100.0%] | $0.0590 |  |
| E4 | gaia_c75 | 20 | 40.0% [20.0%, 60.0%] | $0.0630 |  |
| E4 | gaia_c50 | 20 | 15.0% [0.0%, 30.0%] | $0.0726 |  |
| E4 | isolated_c100 | 20 | 10.0% [0.0%, 25.0%] | $0.0204 |  |
| E4 | isolated_c75 | 20 | 5.0% [0.0%, 15.0%] | $0.0201 |  |
| E4 | isolated_c25 | 20 | 0.0% [0.0%, 0.0%] | $0.0109 |  |
| E4 | gaia_c25 | 20 | 0.0% [0.0%, 0.0%] | $0.0666 |  |
| E4 | isolated_c50 | 20 | 0.0% [0.0%, 0.0%] | $0.0154 |  |
| E8 | gaia_n8 | 20 | 95.0% [85.0%, 100.0%] | $0.0408 |  |
| E8 | gaia_n6 | 20 | 90.0% [75.0%, 100.0%] | $0.0336 |  |
| E8 | gaia_n4 | 20 | 80.0% [60.0%, 95.0%] | $0.0267 |  |
| E8 | gaia_n2 | 20 | 65.0% [45.0%, 85.0%] | $0.0265 |  |
| E8 | homogeneous_n2 | 20 | 15.0% [0.0%, 30.0%] | $0.0156 |  |
| E8 | homogeneous_n4 | 20 | 15.0% [0.0%, 30.0%] | $0.0319 |  |
| E8 | homogeneous_n6 | 20 | 15.0% [0.0%, 30.0%] | $0.0500 |  |
| E8 | homogeneous_n8 | 20 | 15.0% [0.0%, 30.0%] | $0.0501 |  |
| E9 | clean_gaia | 20 | 100.0% [100.0%, 100.0%] | $0.0583 |  |
| E9 | fault_standard | 20 | 100.0% [100.0%, 100.0%] | $0.0603 |  |
| E9 | fault_gaia | 20 | 85.0% [70.0%, 100.0%] | $0.0774 |  |
| E9 | fault_gaia_partial | 20 | 80.0% [60.0%, 95.0%] | $0.0780 |  |