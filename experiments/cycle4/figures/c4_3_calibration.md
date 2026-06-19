# C4-3 — Calibration / reliability (HONEST: lexical proxy non-discriminative)

**Negative methods result (reported straight):** a lexical
assertiveness−hedge proxy does **not** discriminate confidence in terse
mathematical reasoning — mean proxy ≈ 0.50 for *every* role (clean-solver
0.502, misled 0.507, reconciler 0.520, aggregator 0.500). The regex markers
("clearly/maybe/...") rarely fire in symbolic math chains, so the proxy
collapses to its prior. The ECE/accuracy table it produced merely re-encodes
already-known per-role accuracy (clean .95, misled .19) — it is **not** a
valid calibration signal and we do not claim it.

This is itself a small, honest finding: **assertiveness-lexicon calibration
proxies (common in chat-style analyses) are inappropriate for terse symbolic
reasoning** — a methods caveat worth a sentence in the paper.

**Principled replacement → C4-3b (uses C4-1's ELICITED confidence).** C4-1
prompts misled solvers for an explicit `CONFIDENCE: X` (0–1) line. Once C4-1
completes we compute the real reliability diagram + ECE from elicited
confidence vs correctness, answering the dangerous-mode question properly:
*are misled solvers confidently wrong?* (the mechanistic reason
majority/self-consistency fail and a structural reconciler is needed).
Data: `experiments/cycle4/results/c4_1_*.json` (per-sample confidence).
