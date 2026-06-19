# GAIA — Architecture & Code Documentation

This document explains how the GAIA framework is structured and how an episode
runs, so a third party can understand the code before reproducing experiments.
For *running* experiments, see [`REPRODUCE.md`](REPRODUCE.md); for a quick
orientation, see [`README.md`](README.md).

---

## 1. Core idea

GAIA is a **blackboard architecture** for multi-agent LLM collaboration. Instead
of agents talking to each other in a chat transcript, they read from and write to
a single structured **shared workspace** (the blackboard). Coordination emerges
from the state of that workspace, not from a conversation.

The load-bearing mechanism is **conflict-as-task**: when agents disagree (or a
verifier rejects an output), the disagreement is turned into an explicit *fix
task* that is routed to an **independent reconciler**, which re-derives the
answer rather than averaging or voting over the existing ones. This is what lets
GAIA recover *correlated* failures — cases where a majority of agents share the
same wrong answer — which voting and self-consistency cannot.

---

## 2. The blackboard data model (`gaia/blackboard/`)

Everything agents produce is a typed record on the blackboard:

| Object | Meaning |
|---|---|
| **Task** | A unit of work to be claimed and executed |
| **Artifact** | An output produced by an agent (a solution, a code patch, …) |
| **Evidence** | Support attached to an artifact (test results, derivations) |
| **Signal** | A coordination event (e.g. `CONFLICT`, uncertainty) |
| **Policy** | Tunable coordination parameters (used by meta-update) |

- `models.py` — the typed records above.
- `storage.py` — `InMemoryStorage` (default) and `SQLiteStorage` back-ends.
- `blackboard.py` — the coordination hub: post/claim/read/write + an append-only
  audit log that makes every episode fully reconstructable.

Because all state is structured and append-only, every episode is **auditable by
construction** — the full reasoning trace, signals, and timings are recoverable
from the saved state.

---

## 3. Agents (`gaia/agents/`)

Role-specialized agents extend `base.py` (`BaseAgent`), which implements
**self-assignment**: agents poll the blackboard and claim tasks via a **lease**
mechanism that prevents two agents doing the same work. Agents run in two tiers
(Feature B) — fast models for routine production, slower/stronger models for
verification and reconciliation. Domain variants live under `agents/math/`,
`agents/puzzle/`, `agents/miniwob/`.

---

## 4. The episode loop (`gaia/episode/`)

`loop.py` (`EpisodeLoop`) orchestrates a coordination cycle; `scheduler.py`
handles self-assignment. Each episode follows a 7-step cycle:

1. **Initialize** — post the root task to the blackboard.
2. **Decompose** — a planner creates subtasks (optional).
3. **Self-assignment** — agents poll and claim tasks (lease-based).
4. **Production** — agents execute and write artifacts + evidence.
5. **Detect signals** — find conflicts / uncertainty on the board.
6. **Resolve** — conflict-as-task and (optionally) branch-and-merge.
7. **Verify** — check artifacts against acceptance criteria.

Steps 5–7 are where GAIA differs from a vote: a `CONFLICT` signal escalates to
an independent reconciler (step 6) and a verifier gates the result (step 7).

---

## 5. Resolution & meta (`gaia/resolution/`, `gaia/meta/`)

- `resolution/conflict.py` — Feature E: detect disagreement, raise a `CONFLICT`
  signal, spawn a fix task, route it to an independent reconciler.
- `resolution/branch_merge.py` — Feature F: fork the blackboard, explore
  approaches in parallel, merge the best.
- `meta/policy.py`, `meta/meta_update.py` — Feature G: adjust coordination policy
  across episodes from observed outcomes.

---

## 6. Methods, providers, execution

- `methods/` — the five composable configurations (`single_agent`,
  `multi_agent_chat`, `gaia_ae`, `gaia_af`, `gaia_ag`) used as experiment arms.
- `llms/` — a unified `BaseLLM` interface with `openai`, `anthropic`, `groq`,
  and `gemini` back-ends; every call is cost-tracked.
- `execution/` — sandboxed code execution (for code tasks / verification).
- `parsers/`, `prompts/`, `utils/`, `micro_checkers.py` — answer parsing,
  prompt templates, shared utilities, and lightweight structural checks.

---

## 7. Experiments (`experiments/`)

Each experiment is a self-contained directory, typically:

```
experiments/<name>/
  scripts/    # runnable entry points (run_*.py) + analyzers (analyze_*.py)
  results/    # produced result JSON + saved episode state dumps
  figures/    # produced plots
  configs/    # (where applicable) per-arm YAML settings
```

Runs are **seeded and checkpointed** — re-running a command skips completed items
and resumes. The naming follows the paper: `E*` core experiments, `NX*`
baselines/robustness/generalization, `W*` stretch/robustness, `cycle3–5` the
later analysis cycles. [`REPRODUCE.md`](REPRODUCE.md) maps every ID to its exact
command and output location.

---

## 8. Reproducibility & data integrity

- Every full-pipeline episode writes a complete state dump (reasoning chains,
  signals, evidence, phase timings) plus a per-call event log.
- Default models are fixed inside each script (not CLI flags) for determinism.
- Bulky stdout `*.log` files are git-ignored (regenerable); the structured state
  dumps that analyzers consume are retained.
- `.env` (live API keys) is **never** committed — use `.env.example` as the
  template.

---

## 9. Key design decisions

1. **Pure Python** coordination primitives (no heavyweight orchestration
   dependency) — full control over the blackboard and self-assignment.
2. **Lease mechanism** prevents duplicate work in self-assignment.
3. **In-memory storage first** for speed, with an optional SQLite back-end.
4. **Multi-provider** unified interface so the same experiment runs across model
   families (used for the cross-model generalization results).
5. **Cost tracking** on every LLM call for the cost/accuracy analyses.
