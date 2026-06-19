# C3-5 — Semantic reasoning-chain analysis (E3, n_chains=299)

## (1) Do roles reason in a semantically distinct *kind*?
Role-centroid pairwise cosine (1.0=identical kind, lower=distinct):
- aggregator~reconciler: 0.629
- aggregator~solver: 0.355
- reconciler~solver: 0.738

Role separation score (in-role cohesion − nearest other centroid; higher = more distinct reasoning kind):
- aggregator: 0.332  (n=52)
- reconciler: 0.084  (n=52)
- solver: -0.061  (n=195)

## (2) Intra-episode semantic spread vs success
- mean spread | PASSED episodes: 0.1878 (n=13)
- mean spread | FAILED episodes: None (n=0)

## Reading
- If reconciler↔solver centroid cosine is materially below solver↔solver, the reconciler reasons in a *different kind*, not merely longer — upgrading W10's length proxy to a semantic claim and reinforcing the E3 mechanism (the reconciler does a qualitatively different audit).
- Spread vs success links *content diversity* (not char length) to outcome. Honest: embedding-geometry is a proxy for 'reasoning kind'; ~$0.1, fully reproducible (text-embedding-3-small).