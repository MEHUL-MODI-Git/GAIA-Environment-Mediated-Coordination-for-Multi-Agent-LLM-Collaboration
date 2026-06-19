# NX5b — Branch-and-Merge Operating Regime (when does Feature F help?)

## Motivation
NX5 isolated Feature F (branch-and-merge) and found a **clean null**: on a hard
HumanEval slice with a strong solver (`gpt-4.1-mini`, pass@1 = 89%),
branch_on = branch_off = 88.9% — 0 recovered, 0 regressed, at ~3.8× cost. That
result is honest but one-sided: it never tested the regime where Feature F
*should* help. NX5b supplies that regime, so together they **characterize when
branch-and-merge helps** instead of leaving it an unexplained negative.

## Why it should help here (grounded in the code + the literature)
- **Mechanism (from `gaia/resolution/branch_merge.py`):** each branch runs the
  real HumanEval unit tests; merge takes the **first branch that passes the
  tests**. So Feature F = repeated diverse sampling **with an oracle (unit-test)
  selector** ≈ pass@(1+n) realized.
- **Large Language Monkeys** (Brown et al., arXiv:2407.21787): repeated sampling
  scales *coverage* (pass@k) log-linearly and "amplifies weaker models"; and *in
  verifiable domains like coding, coverage gains translate directly into
  performance.*
- **Generation–verification gap** (Weaver, arXiv:2506.18203): an *oracle*
  verifier realizes the coverage gain that weak verifiers/majority-vote cannot.
  GAIA's unit-test selector is that oracle.

⇒ **Hypothesis (pre-registered):** branch-and-merge's benefit is governed by
*headroom* (pass@k − pass@1). With a **weak solver** (real headroom, stochastic
recoverable failures) branch_on ≫ branch_off (recovered ≫ regressed); with a
**strong solver** (NX5 regime, no headroom) the benefit ≈ 0.

## Design
Reuse the NX5 harness (`EpisodeLoop`, `Policy.branch_trigger_on_failure`,
`n_branches=3`). Two factors:

| Factor | Levels |
|---|---|
| Solver model | **WEAK** `gpt-4.1-nano`  ·  STRONG `gpt-4.1-mini` (NX5 replication) |
| Branch | OFF (single trajectory) · ON (n_branches=3) |

- **Slice:** a representative HumanEval slice (not the hardest-only) so the weak
  model's failures include *recoverable* ones. Start n≈30.
- **Stretch:** on the weak model, sweep `n_branches ∈ {1,3,5}` to show monotone
  coverage gain (the inference-time scaling curve).

## Metrics
accuracy (off vs on) · **recovered** (off-fail→on-pass) · **regressed** ·
branches fired · cost (USD) · effective pass@1 vs pass@(1+n) · oracle-coverage
(did *any* branch pass) to confirm the verifier closes the gen–verification gap.

## Success criterion
On the weak solver: **recovered ≫ regressed** and branch_on accuracy
significantly > branch_off (non-overlapping bootstrap CIs at the full n), while
the strong solver reproduces NX5's ≈0 benefit. That demonstrates the *regime*,
not a cherry-picked point.

## Honest bounds (state in any write-up)
- The win is **coverage realized by a perfect (unit-test) selector**; where no
  oracle verifier exists, the gain may not be realizable (cf. Weaver / LLM-
  Monkeys non-verifiable plateau). Feature F is therefore a *verifiable-task*
  mechanism.
- Cost scales ~(1+n)×; report it. The claim is "accuracy-for-compute in the
  recoverable-headroom regime," not a free lunch.

## Protocol
1. Pilot: n=8, both models, off/on — confirm the weak model shows recovery and
   the strong model doesn't, before scaling.
2. Full: n≈30 (+ bootstrap CIs); optional n_branches sweep on the weak model.
3. If positive and clean → candidate to merge back into the main repo as the
   companion to NX5 ("when Feature F helps").
