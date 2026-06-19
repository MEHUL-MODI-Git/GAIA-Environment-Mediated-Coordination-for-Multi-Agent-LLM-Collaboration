# GAIA: Environment-Mediated Coordination for Multi-Agent LLM Collaboration

A research framework for multi-agent LLM systems that coordinate through a
**shared blackboard** rather than conversational debate. GAIA's distinctive
mechanism is **conflict-as-task escalation to an independent reconciler**, which
recovers *correlated* reasoning failures that majority voting and unstructured
compute-scaling cannot.

This repository accompanies the URECA@NTU research project *GAIA: Environment-
Mediated Coordination for Multi-Agent LLM Collaboration*. It contains the full
framework, every experiment in the paper, and a complete reproduction guide.

> **To reproduce the paper's experiments, see [`REPRODUCE.md`](REPRODUCE.md)** —
> the single authoritative list of how to run every experiment, with commands,
> output locations, and approximate costs. This README is the framework
> orientation; `REPRODUCE.md` is the verification contract.
>
> For an architecture walkthrough and the repository map, see
> [`DOCUMENTATION.md`](DOCUMENTATION.md).

---

## What GAIA does

GAIA coordinates role-specialized agents over a structured shared workspace:

- **Blackboard coordination** instead of conversational debate
- **Verification gates** that validate outputs against acceptance criteria
- **Conflict-as-task resolution** — disagreements become explicit fix tasks
  routed to an *independent* reconciler that re-derives the answer
- **Branch-and-merge** for parallel solution exploration
- **Meta-update** for cross-episode policy tuning

### GAIA Features (A–G)

| | Feature | Role |
|---|---|---|
| **A** | Shared Blackboard | Environment-mediated coordination substrate |
| **B** | Agent Tiers | Fast / slow models for different task types |
| **C** | Self-Assignment | Agents poll and claim tasks via leases |
| **D** | Agent Spawning | Dynamic scaling based on backlog |
| **E** | Conflict-as-Task | Failures become explicit fix tasks (→ reconciler) |
| **F** | Branch-and-Merge | Fork the board, try approaches in parallel, merge best |
| **G** | Meta-Update | Learn from episodes to tune coordination policy |

The five experiment **methods** compose these features incrementally:

| Method | Features | Description |
|---|---|---|
| `single_agent` | — | Single LLM with retry loop (baseline) |
| `multi_agent_chat` | Chat | Solver debates a critic until consensus |
| `gaia_ae` | A–E | Blackboard + verification + conflict-as-task |
| `gaia_af` | A–F | + branch-and-merge |
| `gaia_ag` | A–G | + meta-update (full GAIA) |

---

## Installation

Requires **Python 3.10+**.

```bash
cd "GAIA URECA"
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .                                   # installs deps from pyproject.toml
```

**API keys.** Copy `.env.example` to `.env` and fill in real values (the `.env`
file is git-ignored and must never be committed):

```bash
cp .env.example .env
```

| Key | Needed for |
|---|---|
| `OPENAI_API_KEY` | almost all experiments (default provider) |
| `ANTHROPIC_API_KEY` | C5-xmodel, NX2 cross-vendor Claude judge |
| `GROQ_API_KEY` | NX3-open (open-weight Llama), C3-7 steering |

---

## Quick start

The HumanEval method-comparison harness is the simplest entry point. Run it from
the repository root:

```bash
# Method 1: single-agent baseline (first 5 problems)
python experiments/humaneval/scripts/run_experiment.py \
    --method single_agent \
    --data data/humaneval/test.jsonl \
    --output results/method1.jsonl \
    --problems 0-5

# Method 5: full GAIA (A–G)
python experiments/humaneval/scripts/run_experiment.py \
    --method gaia_ag \
    --provider openai --model gpt-4o-mini \
    --data data/humaneval/test.jsonl \
    --output results/method5.jsonl \
    --problems 0-10
```

`run_experiment.py` accepts `--method {single_agent,multi_agent_chat,gaia_ae,
gaia_af,gaia_ag}`, `--provider {openai,anthropic,groq,gemini}`, `--model`,
`--data`, `--output`, and `--problems`. Per-method recommended settings are in
`experiments/humaneval/configs/method{1..5}_*.yaml`.

**For the paper's actual studied results** (correlated-failure recovery, causal
attribution, cross-model generalization, the honest negatives, etc.), follow
[`REPRODUCE.md`](REPRODUCE.md) — the HumanEval harness above is one substrate
among many.

---

## LLM providers

A unified interface wraps four providers; pick fast/slow models per tier:

```bash
--provider openai    --model gpt-4o-mini   # / gpt-4o
--provider anthropic --model claude-3-5-haiku-20241022   # / sonnet
--provider groq      --model llama-3.3-70b-versatile
--provider gemini    --model gemini-1.5-flash   # / pro
```

---

## Repository layout

```
gaia/                 # the framework (importable package)
  blackboard/         # Feature A: Task/Artifact/Evidence/Signal models + storage
  agents/             # Features B,C: role agents (+ math/puzzle/miniwob variants)
  llms/               # multi-provider LLM interface (openai/anthropic/groq/gemini)
  episode/            # the coordination loop + self-assignment scheduler
  resolution/         # Features E,F: conflict-as-task + branch-and-merge
  meta/               # Feature G: policy + meta-update
  methods/            # the 5 experiment methods (single_agent … gaia_ag)
  execution/          # sandboxed code execution
  parsers/  prompts/  utils/   micro_checkers.py
  benchmarks/         # HumanEval / MiniWoB integration
experiments/          # every paper experiment (code + results + figures)
data/                 # datasets, trap suites, puzzles
tests/                # test suite
REPRODUCE.md          # how to run every experiment (verification contract)
DOCUMENTATION.md      # architecture walkthrough + repo map
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the 7-step coordination cycle and
a deeper architecture description.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Copyright

Copyright (c) 2026 Mehul Modi. All rights reserved.

This repository accompanies a URECA@NTU research project and is shared for
review and reference. No open-source license is granted at this time; please
contact the author regarding reuse.
