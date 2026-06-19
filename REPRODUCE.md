# GAIA — Reproduction Guide (for code verification)

**Single authoritative list of how to run every experiment in the paper.**
Every command below was verified against the actual script entry points.

> **MAINTENANCE RULE:** Whenever a *new* experiment is added or run, its
> command + output location **must** be added to the table in this file in the
> same change. This file is the contract with the verifier — keep it complete.

---

## 1. Prerequisites (do this once)

```bash
# Python 3.10+ required
cd /path/to/GAIA
python -m venv venv && source venv/bin/activate
pip install -e .                 # installs all deps from pyproject.toml
```

**API keys** — create a `.env` file in the repo root (see `.env.example`):

```
OPENAI_API_KEY=sk-...            # required for almost all experiments
ANTHROPIC_API_KEY=sk-ant-...     # required only for C5-xmodel + NX2 Claude judge
GROQ_API_KEY=gsk-...             # required only for NX3-open + C3-7 (open-weight)
```

**Rules that apply to every command below:**
- Run **from the repo root** (paths are relative to it).
- Default model is OpenAI `gpt-4.1-nano`/`gpt-4.1`, set inside each script
  (not a CLI flag) — this is intentional and fixed for reproducibility.
- Runs are **seeded and checkpointed**: re-running a command skips already
  completed items and resumes.
- Result JSON/figures are written under `experiments/<name>/results|figures/`.
- Most scripts print a one-line `Saved ...` with the exact output path.

---

## 2. Core paper experiments (E3–E9)

| ID | What it tests | Command(s) | Output |
|----|---------------|------------|--------|
| **E3** | Correlated failure: reconciler beats majority vote | `python experiments/correlated_failure/scripts/run_correlated_failure.py --condition all`<br>then `python experiments/correlated_failure/scripts/analyze_results.py` | `experiments/correlated_failure/results/` |
| **E4** | Info-asymmetry coverage scaling | `python experiments/puzzle/scripts/run_puzzle_coverage.py --coverage all --condition all`<br>then `python experiments/puzzle/scripts/analyze_coverage_scaling.py` | `experiments/puzzle/results/` |
| **E5** | Mechanism: CONFLICT fires on 85% of real tasks (free, reuses HumanEval logs) | `python experiments/humaneval/scripts/analyze_mechanism.py` | `experiments/humaneval/` |
| **E8** | Agent scaling vs role diversity | `python experiments/puzzle/scripts/run_puzzle_scaling_agents.py --num-agents all --condition-type all`<br>then `python experiments/puzzle/scripts/analyze_agent_scaling.py` | `experiments/puzzle/results/` |
| **E9** | Fault injection (honest negative) | `python experiments/fault_injection/scripts/run_fault_injection.py --condition all`<br>then `python experiments/fault_injection/scripts/analyze_fault_injection.py` | `experiments/fault_injection/results/` |

Per-experiment agent-behaviour figures (free, after the run):
```bash
python experiments/viz/analyze_agent_behavior.py --exp correlated_failure
python experiments/viz/analyze_agent_behavior.py --exp fault_injection
python experiments/viz/analyze_agent_behavior.py --exp puzzle --subdir coverage
python experiments/viz/analyze_agent_behavior.py --exp puzzle --subdir scaling
```

---

## 3. NX experiments (baselines, robustness, generalization)

All NX runners take **no arguments** (conditions are fixed inside the script)
unless noted. Run, then run the matching analyzer.

| ID | What it tests | Command(s) |
|----|---------------|------------|
| **NX1** | Real-framework baselines (debate, plain-blackboard) on E3 traps | `python experiments/nx1_baselines/scripts/run_nx1.py`<br>then `python experiments/nx1_baselines/scripts/analyze_nx1.py` |
| **NX9-dose** | GAIA accuracy vs #misled (operating envelope) | `python experiments/nx1_baselines/scripts/run_dose.py` |
| **C2-4** | Strategic-deception agent | `python experiments/nx1_baselines/scripts/run_deception.py` |
| **NX2** | MAST 14-mode failure analysis (multi-step — see §3a) | see §3a below |
| **NX3-open** | Cross-model generalization on open-weight Llama (needs `GROQ_API_KEY`) | `python experiments/nx3_open/scripts/run_nx3_open.py` |
| **NX5** | Branch-and-merge / Feature F (positional arg = #problems) | `python experiments/nx5_branch/scripts/run_nx5.py 18` |
| **NX5b** | Branch-and-merge operating regime — Feature F helps in proportion to solver headroom (paper §5.5: strong solver Δ≈0; weak solver ~+15pts). `--model` sets the pool; weak solver shows the gain | `python experiments/nx5b_branch_regime/scripts/run_nx5b.py --model gpt-3.5-turbo --n 32 --start 2 --step 5`<br>(strong-solver null: `--model gpt-4.1-nano --n 16 --start 20 --step 6`) |
| **NX6** | Visibility-topology sweep (honest negative) | `python experiments/nx6_topology/scripts/run_nx6.py` |
| **NX7** | Realistic scheduling task | `python experiments/nx7_realistic/scripts/gen_scheduling.py` (regenerate data, optional)<br>then `python experiments/nx7_realistic/scripts/run_nx7.py` |
| **NX8** | Prompt-injection robustness | `python experiments/nx8_injection/scripts/run_nx8.py` |
| **NX11** | Feature-G meta-update replay (free) | `python experiments/nx11_metaupdate/scripts/run_nx11.py` |

### 3a. NX2 — MAST failure-mode analysis (multi-step)

NX2 classifies failed traces with an LLM judge, then checks judge agreement.
`mast_classifier.py` requires CLI args:

```bash
# Classify failed traces for one system (repeat per system label).
python experiments/nx2_mast/scripts/mast_classifier.py \
    --glob 'experiments/correlated_failure/logs/**/*.state.json' \
    --label GAIA --max 200
# Cross-vendor judge agreement (uses ANTHROPIC_API_KEY for the Claude judge):
python experiments/nx2_mast/scripts/mast_judge_claude.py
python experiments/nx2_mast/scripts/judge_agreement.py
python experiments/nx2_mast/scripts/analyze_nx2.py
```

---

## 4. W experiments (stretch / robustness)

No-arg runners. Convenience batch scripts run them sequentially:

```bash
zsh experiments/run_queue.sh      # NX5, NX8, W1 (one at a time)
zsh experiments/run_queue2.sh     # W6, W9, W3 (one at a time)
```

Or individually:

| ID | What it tests | Command |
|----|---------------|---------|
| **W1** | Chaos / agent-dropout resilience | `python experiments/w1_chaos/scripts/run_w1.py` |
| **W3** | Emergent specialization (honest negative) | `python experiments/w3_emergent/scripts/run_w3.py` |
| **W6** | Blackboard-primitive ablation | `python experiments/w6_primitive/scripts/run_w6.py` |
| **W9** | MAST failure-injection matrix | `python experiments/w9_inject/scripts/run_w9.py` |

---

## 5. Cycle-3 — emergence & governance (free, re-mines existing dumps)

```bash
python experiments/cycle3/scripts/emergence_pid.py     # C3-1 emergence engine
python experiments/cycle3/scripts/dump_analytics.py    # C3-2/3/6 traceability
python experiments/cycle3/scripts/run_c3_5.py          # C3-5 semantic chains
python experiments/cycle3/scripts/run_c3_7.py          # C3-7 steering (needs GROQ_API_KEY)
```

---

## 6. Cycle-4 — compute-matched baselines & causal attribution

Run the experiment, then its analyzer. The OpenAI variants are the canonical
runs reported in the paper (`_openai` suffix).

```bash
python experiments/cycle4/scripts/run_c4_1_openai.py   # C4-1 compute-matched
python experiments/cycle4/scripts/analyze_c4_1.py      # → c4_1 frontier + c4_3b calib

python experiments/cycle4/scripts/run_c4_2.py          # C4-2 behavioral taxonomy
python experiments/cycle4/scripts/run_c4_3.py          # C4-3 calibration (honest negative)
python experiments/cycle4/scripts/run_c4_4_openai.py   # C4-4 leave-one-out
python experiments/cycle4/scripts/run_c4_5_openai.py   # C4-5 do-operator
python experiments/cycle4/scripts/analyze_c4_45.py     # → c4_45 causal decomposition
python experiments/cycle4/scripts/run_c4_6.py          # C4-6 invariant-checker
```
(`run_c4_1.py`, `run_c4_4.py`, `run_c4_5.py` without `_openai` are the earlier
Groq attempts — kept for provenance, **not** the canonical runs.)

---

## 7. Cycle-5 — frontier-grounded rigor

Generate the expanded verified traps first (one-time), then run experiments,
then build figures.

```bash
python experiments/cycle5/scripts/gen_traps.py         # → data/gsm8k/correlated_failure_problems_expanded.json
python experiments/cycle5/scripts/run_c5_1.py          # C5-1 token-budget-matched
python experiments/cycle5/scripts/run_c5_1b.py         # C5-1b biased-self-critique
python experiments/cycle5/scripts/run_c5_2.py          # C5-2 exact Shapley
python experiments/cycle5/scripts/run_c5_3.py          # C5-3 info-controlled isolation
python experiments/cycle5/scripts/run_c5_4.py          # C5-4 semantic-drift (honest null)
python experiments/cycle5/scripts/run_c5_5.py          # C5-5 reconciler bottleneck
python experiments/cycle5/scripts/run_c5_5b.py         # C5-5b stress-test
python experiments/cycle5/scripts/run_c5_xmodel.py     # cross-model (needs ANTHROPIC_API_KEY)

# Figures (run after the matching experiment; these have no CLI — run directly):
python experiments/cycle5/scripts/fig_c5_1.py
python experiments/cycle5/scripts/fig_c5_2.py
python experiments/cycle5/scripts/fig_c5_34.py
python experiments/cycle5/scripts/fig_c5_5.py
python experiments/cycle5/scripts/fig_c5_5b.py
```

---

## 8. Cross-experiment analysis & poster figures (free, no API)

Run after the underlying experiments exist:

```bash
python experiments/viz/cross_experiment_analysis.py     # NX4 Pareto + bootstrap CIs
python experiments/viz/coordination_fingerprint.py      # W10 fingerprint PCA
python experiments/viz/information_flow.py               # CONFLICT→Reconciler routing
python experiments/viz/diversity_decomposition.py        # C2-1 diversity theorem
python experiments/viz/interaction_graph_centrality.py   # C2-3 centrality
python experiments/viz/causal_mediation.py               # C2-2 controlled effect
python experiments/viz/scaling_and_latency.py            # W4 scaling law + W8 wall-clock
python experiments/viz/generate_poster_figures.py        # poster figure set
python experiments/viz/visualize_blackboard_trace.py     # qualitative blackboard trace
python experiments/viz/visualize_conflict_trace.py       # qualitative conflict trace
```

---

## 9. Legacy / benchmark harness (not paper-headline, but runnable)

```bash
# Original HumanEval method1-5 harness (the README documents this interface):
python experiments/humaneval/scripts/run_experiment.py --method gaia_ag \
    --data data/humaneval/test.jsonl --output results/m5.jsonl --problems 0-10
# GSM8K, MiniWoB, base puzzle:
python experiments/gsm8k/scripts/run_gsm8k.py --n_problems 50
python experiments/miniwob/scripts/run_miniwob.py --difficulty easy
python experiments/puzzle/scripts/run_puzzle.py --condition all
```

---

## 10. Repository scope (what this clean submission includes / excludes)

This is the cleaned verification submission. It contains the framework
(`gaia/`), every paper experiment (`experiments/`, with code + result JSON +
retained episode **state dumps** + figures), datasets (`data/`), tests
(`tests/`), and docs (`README.md`, `DOCUMENTATION.md`, this file, `LICENSE`).

The following were intentionally **excluded** and are not needed to reproduce
any experiment:

- **`.env`** — live API keys. Use `.env.example` (provided) as the template.
- **Bulky stdout `*.log` files** (e.g. a 137 MB HumanEval run log) — these are
  regenerable console output. The structured `*.jsonl` state dumps that the
  analyzers actually consume (e.g. E5's `analyze_mechanism.py`) **are retained**,
  so every analysis still runs.
- **Prototype/scratch archive** and old AgentVerse code — not part of the paper.
- **Paper drafts and internal working notes** — the paper is submitted
  separately; this repository is the code artifact.

---

## 11. Approximate costs (USD, for planning)

E3 ≈ $0.36 · E4 ≈ $6.6 · E8 ≈ $5.5 · E9 ≈ $5.5 · E5 free. NX/W/C each
≈ $0.2–$2. Free (no API): E5, NX11, all of Cycle-3, C4-6, and everything in
§8. Re-running is cheaper — checkpoints skip completed items.
