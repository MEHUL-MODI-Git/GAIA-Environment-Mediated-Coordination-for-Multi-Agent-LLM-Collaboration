"""MiniWoB++ structured episode logger

Wraps BlackboardLogger and adds browser-specific events:
  - Every browser action (CLICK/TYPE/SELECT/PRESS_KEY) with element details
  - DOM state snapshot before each action
  - Reward and success result after each action
  - Conflict signals, Critic feedback, strategy resets
  - Full episode summary (tokens, cost, time, action trace)

Output per task:
  logs/{task_name}/episode.jsonl    — machine-readable event stream
  logs/{task_name}/episode.txt      — human-readable step-by-step log

Output aggregate:
  logs/run_events.jsonl             — all events from the full run (append mode)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .blackboard_logger import BlackboardLogger, EventType


# Extra event types for MiniWoB++
_BROWSER_STEP = "browser_step"
_DOM_SNAPSHOT = "dom_snapshot"
_STRATEGY_RESET = "strategy_reset"
_ACTION_DECIDED = "action_decided"
_EPISODE_METRICS = "episode_metrics"


class MiniWoBLogger:
    """Structured logger for one MiniWoB++ run.

    One instance per task episode.  Wraps BlackboardLogger for blackboard
    events and adds browser-specific logging on top.
    """

    def __init__(
        self,
        task_id: str,
        log_dir: Path,
        run_log_path: Optional[Path] = None,
        log_to_console: bool = False,
    ):
        """
        Args:
            task_id:       e.g. "miniwob/click-button"
            log_dir:       Directory for this task's logs (created if needed)
            run_log_path:  Shared append-mode log for the entire run (optional)
            log_to_console: Whether to print events to stdout
        """
        self.task_id = task_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / "episode.jsonl"
        self.txt_path = self.log_dir / "episode.txt"
        self.run_log_path = run_log_path

        self._start_time = time.time()
        self._events: List[Dict[str, Any]] = []

        # Accumulators for the episode summary
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_llm_calls = 0
        self._browser_steps: List[Dict] = []
        self._conflicts = 0
        self._strategy_resets = 0

        # Hand off blackboard-level logging to BlackboardLogger
        self.bb_logger = BlackboardLogger(
            log_file=self.jsonl_path,
            log_to_console=log_to_console,
        )

        # Write episode header to .txt
        self._txt_lines: List[str] = []
        self._txt("=" * 64)
        self._txt(f"Task:        {task_id}")
        self._txt(f"Start:       {datetime.utcnow().isoformat()} UTC")
        self._txt("=" * 64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _txt(self, line: str):
        """Append a line to the human-readable log buffer."""
        self._txt_lines.append(line)

    def _flush_txt(self):
        """Write all buffered text lines to the .txt file."""
        with open(self.txt_path, "w") as f:
            f.write("\n".join(self._txt_lines) + "\n")

    def _elapsed(self) -> str:
        """Elapsed time as mm:ss.f string."""
        s = time.time() - self._start_time
        m = int(s // 60)
        return f"{m:02d}:{s % 60:04.1f}"

    def _write_event(self, event: Dict[str, Any]):
        """Append one JSON event to the per-task JSONL and run log."""
        line = json.dumps(event, default=str)
        with open(self.jsonl_path, "a") as f:
            f.write(line + "\n")
        if self.run_log_path:
            with open(self.run_log_path, "a") as f:
                f.write(line + "\n")

    def _make_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "task_id": self.task_id,
            "event": event_type,
            **kwargs,
        }

    # ------------------------------------------------------------------
    # Blackboard logger proxy (used by Blackboard & agents automatically)
    # ------------------------------------------------------------------

    @property
    def blackboard_logger(self) -> BlackboardLogger:
        """Return the wrapped BlackboardLogger for wiring into Blackboard."""
        return self.bb_logger

    # ------------------------------------------------------------------
    # Episode-level events
    # ------------------------------------------------------------------

    def log_episode_start(self, instruction: str):
        ev = self._make_event("episode_start", instruction=instruction)
        self._write_event(ev)
        self._txt(f"[{self._elapsed()}] EPISODE START")
        self._txt(f"           Instruction: {instruction}")
        self._flush_txt()

    # ------------------------------------------------------------------
    # Browser action events (called from web_loop.py)
    # ------------------------------------------------------------------

    def log_action_decided(
        self,
        agent_name: str,
        action: Dict[str, Any],
        step: int,
        iteration: int,
    ):
        """Log what action the Navigator decided (before executing it)."""
        action_type = action.get("type", "?")
        element = action.get("element_desc", action.get("ref", ""))

        ev = self._make_event(
            _ACTION_DECIDED,
            agent=agent_name,
            step=step,
            iteration=iteration,
            action_type=action_type,
            element=element,
            action=action,
        )
        self._write_event(ev)
        self._txt(f"[{self._elapsed()}] STEP {step} decide: {action_type} {element!r}")
        self._flush_txt()

    def log_browser_step(
        self,
        step: int,
        action: Dict[str, Any],
        reward: float,
        success: bool,
        done: bool,
        dom_elements: List[Dict],
    ):
        """Log the result of one browser action."""
        action_type = action.get("type", "?")
        element = action.get("element_desc", action.get("ref", ""))

        ev = self._make_event(
            _BROWSER_STEP,
            step=step,
            action_type=action_type,
            element=element,
            action=action,
            reward=reward,
            success=success,
            done=done,
            dom_element_count=len(dom_elements),
        )
        self._write_event(ev)
        self._browser_steps.append(ev)

        status = "✓ SUCCESS" if success else ("DONE" if done else "→")
        self._txt(
            f"[{self._elapsed()}] STEP {step:2d}: {action_type} {element!r}"
            f"  reward={reward:.2f}  {status}"
        )
        self._flush_txt()

    def log_dom_snapshot(self, step: int, elements: List[Dict]):
        """Log current DOM state (condensed — tag + text + ref only)."""
        condensed = [
            {"tag": el.get("tag"), "text": el.get("text", "")[:40], "ref": el.get("ref")}
            for el in elements
            if el.get("text") or el.get("tag") in ("button", "input", "input_text", "input_checkbox", "input_radio", "input_number", "select", "a", "textarea", "option", "t")
        ]
        ev = self._make_event(_DOM_SNAPSHOT, step=step, elements=condensed)
        self._write_event(ev)

    def log_conflict(self, step: int, conflict_count: int):
        """Log a CONFLICT signal being raised."""
        self._conflicts += 1
        ev = self._make_event("conflict_raised", step=step, total_conflicts=conflict_count)
        self._write_event(ev)
        self._txt(f"[{self._elapsed()}]   ⚡ CONFLICT raised (step {step}, total={conflict_count})")
        self._flush_txt()

    def log_critic_feedback(self, step: int, feedback: str):
        """Log Critic's feedback after reviewing stuck actions."""
        ev = self._make_event("critic_feedback", step=step, feedback=feedback[:300])
        self._write_event(ev)
        short = feedback[:100].replace("\n", " ")
        self._txt(f"[{self._elapsed()}]   🔍 CRITIC: {short}")
        self._flush_txt()

    def log_strategy_reset(self, step: int, critic_fire_count: int, failure_summary: str):
        """Log a strategy reset (branch-and-merge equivalent)."""
        self._strategy_resets += 1
        ev = self._make_event(
            _STRATEGY_RESET,
            step=step,
            critic_fires=critic_fire_count,
            failure_summary=failure_summary[:200],
        )
        self._write_event(ev)
        self._txt(f"[{self._elapsed()}]   🔄 STRATEGY RESET (critic fired {critic_fire_count}x) — Planner will replan")
        self._flush_txt()

    def log_llm_call(
        self,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ):
        """Mirror LLM call into the structured log with token + cost details."""
        total = prompt_tokens + completion_tokens
        self._total_tokens += total
        self._total_cost += cost_usd
        self._total_llm_calls += 1

        ev = self._make_event(
            "llm_call",
            agent=agent_name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 1),
        )
        self._write_event(ev)
        self._txt(
            f"[{self._elapsed()}]   LLM {agent_name} → {model} "
            f"({total} tok, ${cost_usd:.5f}, {latency_ms/1000:.1f}s)"
        )
        self._flush_txt()

    # ------------------------------------------------------------------
    # Episode end
    # ------------------------------------------------------------------

    def log_episode_end(
        self,
        passed: bool,
        steps_taken: int,
        action_history: List[str],
        conflicts: int,
        strategy_resets: int,
    ):
        """Write final summary event and flush the human-readable log."""
        duration_s = round(time.time() - self._start_time, 2)

        summary = {
            "passed": passed,
            "steps_taken": steps_taken,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "total_llm_calls": self._total_llm_calls,
            "conflicts": conflicts,
            "strategy_resets": strategy_resets,
            "duration_s": duration_s,
        }
        ev = self._make_event(_EPISODE_METRICS, **summary)
        self._write_event(ev)

        # Human-readable summary footer
        self._txt("")
        self._txt("=" * 64)
        result_str = "✓ PASSED" if passed else "✗ FAILED"
        self._txt(f"RESULT: {result_str}")
        self._txt(f"  Steps:          {steps_taken}")
        self._txt(f"  LLM calls:      {self._total_llm_calls}")
        self._txt(f"  Total tokens:   {self._total_tokens:,}")
        self._txt(f"  Total cost:     ${self._total_cost:.5f}")
        self._txt(f"  Duration:       {duration_s:.1f}s")
        self._txt(f"  Conflicts:      {conflicts}")
        self._txt(f"  Strategy resets:{strategy_resets}")
        if action_history:
            self._txt("")
            self._txt("Action trace:")
            for i, a in enumerate(action_history, 1):
                self._txt(f"  {i:2d}. {a}")
        self._txt("=" * 64)
        self._flush_txt()

        return summary

    def get_summary(self) -> Dict[str, Any]:
        """Return current episode metrics dict (can be called any time)."""
        return {
            "task_id": self.task_id,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "total_llm_calls": self._total_llm_calls,
            "browser_steps": len(self._browser_steps),
            "conflicts": self._conflicts,
            "strategy_resets": self._strategy_resets,
            "duration_s": round(time.time() - self._start_time, 2),
        }
