# NX11 — Feature G (meta-update) mechanism demonstration

Replayed **93** real episode outcomes (E4 coverage gaia + E3 gaia) through MetaUpdater (update_frequency=10) from a deliberately suboptimal start (branching OFF, max_iter=10, spawn_threshold=3).

- Policy updates triggered: **6**
- Final policy: branch_trigger_on_failure=True, max_iterations=19, spawn_threshold=2

## Change log
- after ep 30: max_iterations: 10→12
- after ep 40: max_iterations: 12→14
- after ep 50: max_iterations: 14→16
- after ep 60: max_iterations: 16→18
- after ep 80: max_iterations: 18→20
- after ep 90: spawn_threshold: 3→2, branch_trigger_on_failure: False→True, max_iterations: 20→19

**Takeaway.** Feature G's cross-episode heuristics fire correctly on real metrics: a conflict-heavy non-stationary stream drives the updater to ENABLE branch-and-merge and adjust iteration/spawn caps it had wrongly set — i.e. GAIA can re-tune its own coordination policy from experience. Scope (honest): this demonstrates the adaptation MECHANISM on replayed outcomes; a live closed-loop A/B (adapted-policy accuracy vs frozen) is the stronger follow-up and is left as future work, not claimed here.