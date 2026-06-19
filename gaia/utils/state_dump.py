"""Full blackboard state persistence for post-hoc analysis.

The JSONL event log captures *what happened when* (every agent execution,
LLM call with tokens/cost/latency, task transition, signal, artifact-posted
event). But it does NOT store the full *content* of artifacts — the actual
reasoning chains, deductions, synthesizer outputs, reconciler diagnoses, and
trust tables that agents produced.

`dump_episode_state` serializes the COMPLETE end-of-episode blackboard so any
experiment can be reconstructed and analyzed offline:

  - tasks      : every task + metadata + status transitions
  - artifacts  : every artifact WITH full `content` (the reasoning text) and
                 metadata (answers, trust scores, partition, is_misled, etc.)
  - signals    : every signal WITH description + metadata (conflict pairs,
                 trust scores, severity)
  - evidence   : verification results (passed/failed, proposed solution)
  - claims     : agent assertions + confidence

This is what makes "study agent behaviours and pipeline states" possible:
the dump is a faithful, lossless record of the multi-agent interaction.

Output: <out_dir>/<episode_id>.state.json  (sibling of the .jsonl log)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _model_to_dict(obj: Any) -> Any:
    """Best-effort conversion of a pydantic model / datetime to JSON-safe data."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def dump_episode_state(
    blackboard,
    out_dir: Path,
    episode_id: str,
    condition: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize the full blackboard state for one finished episode.

    Args:
        blackboard: the Blackboard instance used for this episode.
        out_dir: directory to write the dump into (created if missing).
        episode_id: problem/puzzle id (used in the filename).
        condition: experimental condition label (stored in the dump).
        extra: optional extra dict merged into the top level (e.g. result
            summary, ground truth, agent roster).

    Returns:
        Path to the written .state.json file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = blackboard.snapshot()
    storage = snap.get("storage", {})

    def _collection(name: str) -> Dict[str, Any]:
        coll = storage.get(name, {})
        return {k: _model_to_dict(v) for k, v in coll.items()}

    state = {
        "episode_id": episode_id,
        "condition": condition,
        "tasks": _collection("tasks"),
        "artifacts": _collection("artifacts"),
        "signals": _collection("signals"),
        "evidence": _collection("evidence"),
        "claims": _collection("claims"),
        "policy": snap.get("policy", {}),
        "audit_log": snap.get("audit_log", []),
        "counts": {
            "n_tasks": len(storage.get("tasks", {})),
            "n_artifacts": len(storage.get("artifacts", {})),
            "n_signals": len(storage.get("signals", {})),
            "n_evidence": len(storage.get("evidence", {})),
            "n_claims": len(storage.get("claims", {})),
        },
    }
    if extra:
        state["extra"] = extra

    safe_id = episode_id.replace("/", "_")
    suffix = f"_{condition}" if condition else ""
    out_path = out_dir / f"{safe_id}{suffix}.state.json"
    with open(out_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    return out_path


def auto_dump_episode_state(
    blackboard,
    episode_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Dump full state next to the blackboard's JSONL log, deriving the path.

    Writes <jsonl-stem>.state.json in the same directory as the episode's
    JSONL log. Used inside episode loops so every run captures full agent
    behaviour + pipeline state with zero runner changes. Silently no-ops if
    the blackboard has no log file (e.g. unit tests). Never raises — analysis
    persistence must not break an experiment run.
    """
    try:
        log_file = getattr(getattr(blackboard, "logger", None), "log_file", None)
        if not log_file:
            return None
        log_file = Path(log_file)
        out_dir = log_file.parent
        stem = log_file.stem  # e.g. "trap_rate_001_gaia"
        out_path = out_dir / f"{stem}.state.json"

        snap = blackboard.snapshot()
        storage = snap.get("storage", {})

        def _coll(name: str) -> Dict[str, Any]:
            return {k: _model_to_dict(v) for k, v in storage.get(name, {}).items()}

        state = {
            "episode_id": episode_id,
            "tasks": _coll("tasks"),
            "artifacts": _coll("artifacts"),
            "signals": _coll("signals"),
            "evidence": _coll("evidence"),
            "claims": _coll("claims"),
            "policy": snap.get("policy", {}),
            "audit_log": snap.get("audit_log", []),
            "counts": {
                "n_tasks": len(storage.get("tasks", {})),
                "n_artifacts": len(storage.get("artifacts", {})),
                "n_signals": len(storage.get("signals", {})),
                "n_evidence": len(storage.get("evidence", {})),
                "n_claims": len(storage.get("claims", {})),
            },
        }
        if extra:
            state["extra"] = extra
        with open(out_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return out_path
    except Exception:
        return None
