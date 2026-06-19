# C2-3 — Temporal interaction-graph centrality

n=331 GAIA episodes. Mean normalized betweenness per role (share of shortest information paths passing through it):

| role | mean betweenness share | episodes present |
|---|---|---|
| expert | 0.438 | 240 |
| synthesizer | 0.208 | 240 |
| solver | 0.126 | 91 |
| aggregator | 0.096 | 91 |
| CONFLICT | 0.076 | 101 |
| auditor | 0.044 | 40 |
| reconciler | 0.013 | 55 |
| OUTCOME | 0.000 | 331 |

**Interpretation (matches the data, corpus = 305 episodes, mostly puzzle E4/E8/E9).** Despite every agent having full blackboard *visibility*, information *influence* is highly concentrated: **expert+synthesizer carry ≈65% of all betweenness**. The synthesizer is the mandatory integration funnel — every expert deduction must pass through it — while the CONFLICT/reconciler path is a *secondary, on-demand* gate (low mean betweenness because conflict is rare in the puzzle corpus, but decisive when it fires — cf. E3 where 12/13 route through it). Net: GAIA is structurally a **dense-visibility, sparse-influence** system — a few mandatory funnel nodes (synthesizer; conflict-gate when triggered) bound how much any single agent's output can sway the result. This is the mechanism by which a fully-connected blackboard avoids the error-amplification the topology literature attributes to dense graphs (DEEP_ANALYSIS.md §3).