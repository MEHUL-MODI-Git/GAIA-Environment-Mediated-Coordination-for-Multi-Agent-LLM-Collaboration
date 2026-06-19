"""MiniWoB++ episode loop — parallel to loop.py but for web interaction tasks"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..blackboard.blackboard import Blackboard
from ..blackboard.models import Task, ArtifactType, Policy, SignalType
from ..agents.base import BaseAgent
from ..utils.metrics import MetricsCollector
from ..utils.logging import get_logger
from ..utils.budget_monitor import BudgetMonitor
from ..utils.miniwob_logger import MiniWoBLogger
from ..execution.web_runner import WebRunner, WebAction
from pydantic import BaseModel

logger = get_logger("web_episode")


class WebEpisodeResult(BaseModel):
    """Result from running one MiniWoB++ episode"""

    task_id: str
    passed: bool
    steps_taken: int = 0
    total_reward: float = 0.0
    artifacts_created: int = 0
    conflicts_detected: int = 0
    metadata: Dict[str, Any] = {}


def _format_elements(elements) -> str:
    """Format DOMElement list into text for agents"""
    if not elements:
        return "(no elements)"
    lines = []
    for el in elements:
        line = f'[{el.tag}] "{el.text}"'
        if el.ref:
            line += f" ref={el.ref}"
        lines.append(line)
    return "\n".join(lines)


_INTERACTIVE_TAGS = {
    "button", "input", "input_text", "input_checkbox", "input_radio",
    "input_number", "input_date", "select", "a", "textarea", "option",
}


def _resolve_element_ref(action_dict: dict, elements) -> int:
    """Find element ref by matching description to element text/tag.

    Returns the integer ref of the best-matching element, or 0 if not found.
    """
    desc = action_dict.get("element_desc", "").lower().strip().strip("\"'")
    if not desc or not elements:
        return int(action_dict.get("ref", 0) or 0)

    # Pass 1a: exact text match (highest priority — avoids "1" matching "14")
    for el in elements:
        el_text = el.text.lower().strip()
        ref_val = int(el.ref) if str(el.ref).lstrip("-").isdigit() else 0
        if ref_val <= 0:
            continue
        if el_text and el_text == desc:
            return ref_val

    # Pass 1b: substring match (both sides non-empty)
    for el in elements:
        el_text = el.text.lower().strip()
        ref_val = int(el.ref) if str(el.ref).lstrip("-").isdigit() else 0
        if ref_val <= 0:
            continue
        if el_text and (desc in el_text or el_text in desc):
            return ref_val

    # Pass 2: tag keyword in description (e.g. "button", "input", "textbox", "date")
    tag_aliases = {
        "textbox": "input_text", "checkbox": "input_checkbox",
        "text input": "input_text", "text box": "input_text",
        "date": "input_date", "date input": "input_date",
        "date field": "input_date", "date input field": "input_date",
    }
    mapped_tag = tag_aliases.get(desc, desc)
    for el in elements:
        if el.tag.lower() == mapped_tag or el.tag.lower() in desc:
            return int(el.ref)

    # Pass 3: any interactive element as last resort
    for el in elements:
        if el.tag in _INTERACTIVE_TAGS:
            return int(el.ref)

    return 0


class WebEpisodeLoop:
    """Episode loop for MiniWoB++ web interaction tasks.

    Orchestrates: WebPlanner → DOMAnalyzer → WebNavigator → WebVerifier/WebCritic
    Interleaves agent decisions with real browser steps via WebRunner.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        agents: List[BaseAgent],
        metrics: MetricsCollector,
        policy: Policy,
        budget_monitor: Optional[BudgetMonitor] = None,
        miniwob_logger: Optional[MiniWoBLogger] = None,
    ):
        self.blackboard = blackboard
        self.agents = agents
        self.metrics = metrics
        self.policy = policy
        self.budget_monitor = budget_monitor or BudgetMonitor(
            max_cost_per_problem=1.00,
            max_iterations=policy.max_iterations,
            max_llm_calls=100,
        )
        self.bb_logger = blackboard.logger
        self.mw_logger = miniwob_logger

    async def run_episode(self, task_spec: Dict[str, Any]) -> WebEpisodeResult:
        """Run one MiniWoB++ episode.

        Args:
            task_spec: Task definition dict with keys:
                - task_id: str  e.g. "miniwob/click-button"
                - task_name: str  e.g. "click-button"
                - instruction: str  (natural language)
                - max_steps: int
                - difficulty: str  (optional)

        Returns:
            WebEpisodeResult with outcome and metrics
        """
        task_id = task_spec["task_id"]
        task_name = task_spec.get("task_name", task_id.split("/")[-1])
        instruction = task_spec.get("instruction", "")
        max_steps = task_spec.get("max_steps", 15)

        start_time = datetime.utcnow()
        self.bb_logger.log_episode_start(problem_id=task_id)
        logger.info(f"=== Starting web episode: {task_id} ===")
        self.budget_monitor.reset()
        # Step 1: Initialize browser environment
        runner = WebRunner(task_name=task_name, max_steps=max_steps)
        try:
            obs = await runner.reset()
        except Exception as e:
            logger.error(f"Failed to initialize WebRunner for {task_name}: {e}")
            return WebEpisodeResult(
                task_id=task_id, passed=False,
                metadata={"error": str(e), "stop_reason": "env_init_failed"}
            )

        # Update instruction from environment if available
        if obs.instruction:
            instruction = obs.instruction

        if self.mw_logger:
            self.mw_logger.log_episode_start(instruction)

        # Step 2: Post root task to blackboard
        root_task = Task(
            title=f"Web task: {task_name}",
            description=instruction,
            acceptance_criteria="success_flag == True from environment",
            metadata={
                "task_type": "web_interaction",
                "task_id_miniwob": task_id,
                "task_name": task_name,
                "instruction": instruction,
                "max_steps": max_steps,
                "current_step": 0,
                "success_flag": False,
                "action_history": [],
                "raw_html": obs.raw_html,
                "dom_elements": [
                    {
                        "tag": el.tag, "text": el.text,
                        "ref": el.ref, **el.attrs,
                    }
                    for el in obs.elements
                ],
            },
        )
        self.blackboard.post_task(root_task)
        logger.info(f"Posted root task {root_task.task_id} | instruction: {instruction[:80]}")

        total_reward = 0.0
        conflicts_detected = 0
        strategy_resets = 0       # branch-and-merge equivalent: max 1 full strategy reset
        critic_fire_count = 0     # track how many times Critic has run
        done = False
        last_executed_artifact_id: Optional[str] = None  # avoid re-executing stale actions

        # ===== Main episode loop =====
        for iteration in range(self.policy.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1} ---")
            self.metrics.record_iteration()

            # Budget check
            can_continue = self.budget_monitor.record_iteration()
            if not can_continue:
                logger.error("Budget iteration limit exceeded")
                break

            # Step 3: Run agents in parallel (plan, analyze DOM, etc.)
            await self._run_agents_parallel()

            # Step 4: Get latest action from navigator
            action_artifact = self.blackboard.get_latest_artifact(
                root_task.task_id, ArtifactType.CODE
            )
            if not action_artifact:
                logger.info("No action artifact yet, continuing")
                continue

            # Only execute if this is a NEW artifact (not the same one we already ran)
            if action_artifact.artifact_id == last_executed_artifact_id:
                logger.info("Action artifact unchanged — navigator did not run this iteration, skipping browser step")
                continue

            # Parse action from navigator
            try:
                action_dict = json.loads(action_artifact.content)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Could not parse action: {action_artifact.content[:100]}")
                continue

            action_type = action_dict.get("type", "NONE")
            if action_type in ("NONE", "UNKNOWN"):
                logger.info(
                    f"Navigator returned {action_type}, skipping step"
                    + (f" | raw: {action_dict.get('raw', '')[:100]}" if action_type == "UNKNOWN" else "")
                )
                last_executed_artifact_id = action_artifact.artifact_id
                continue

            # Resolve element reference by matching description to actual DOM
            if not action_dict.get("ref") and action_dict.get("element_desc"):
                action_dict["ref"] = _resolve_element_ref(action_dict, obs.elements)

            # Log decided action (before executing)
            current_step = root_task.metadata.get("current_step", 0)
            if self.mw_logger:
                self.mw_logger.log_action_decided(
                    agent_name=action_artifact.author,
                    action=action_dict,
                    step=current_step,
                    iteration=iteration,
                )

            # Step 5: Execute action in browser
            raw_ref = action_dict.get("ref", 0)
            web_action = WebAction(
                type=action_type,
                ref=int(raw_ref) if raw_ref else 0,
                text=action_dict.get("text", ""),
                key=action_dict.get("key", ""),
                value=action_dict.get("value", ""),
            )

            try:
                obs, done = await runner.step(web_action)
            except Exception as e:
                logger.error(f"Browser step failed: {e}")
                done = True
                break

            last_executed_artifact_id = action_artifact.artifact_id
            total_reward += obs.reward
            action_history = root_task.metadata.get("action_history", [])
            if action_type == "TYPE":
                action_str = f"TYPE '{action_dict.get('text','')}' INTO {action_dict.get('element_desc', action_dict.get('ref', ''))}"
            elif action_type == "PRESS_KEY":
                action_str = f"PRESS_KEY {action_dict.get('key','')}"
            else:
                action_str = f"{action_type} {action_dict.get('element_desc', action_dict.get('ref', ''))}"
            action_history.append(action_str)

            # Update root task metadata with new DOM state
            root_task.metadata.update({
                "current_step": obs.step,
                "success_flag": obs.success,
                "interaction_log": "\n".join(runner._interaction_log),
                "raw_html": obs.raw_html,
                "dom_elements": [
                    {
                        "tag": el.tag, "text": el.text, "ref": el.ref,
                        **{k: v for k, v in el.attrs.items() if k in ("id", "classes")},
                    }
                    for el in obs.elements
                ],
                "action_history": action_history,
            })
            self.blackboard.update_task(root_task)

            logger.info(f"Step {obs.step}: {action_type} → reward={obs.reward:.2f} success={obs.success}")

            # Log browser step result
            if self.mw_logger:
                self.mw_logger.log_browser_step(
                    step=obs.step,
                    action=action_dict,
                    reward=obs.reward,
                    success=obs.success,
                    done=done,
                    dom_elements=[
                        {"tag": el.tag, "text": el.text, "ref": el.ref}
                        for el in obs.elements
                    ],
                )
                self.mw_logger.log_dom_snapshot(
                    step=obs.step,
                    elements=[
                        {"tag": el.tag, "text": el.text, "ref": el.ref}
                        for el in obs.elements
                    ],
                )

            # Step 6: Check success
            if obs.success:
                logger.info(f"Task {task_id} PASSED at step {obs.step}!")
                self.blackboard.complete_task(root_task.task_id, [])
                self.bb_logger.log_episode_end(problem_id=task_id, passed=True, code="")
                if self.mw_logger:
                    self.mw_logger.log_episode_end(
                        passed=True,
                        steps_taken=obs.step,
                        action_history=root_task.metadata.get("action_history", []),
                        conflicts=conflicts_detected,
                        strategy_resets=strategy_resets,
                    )
                passed, log = runner.result()
                runner.close()
                return WebEpisodeResult(
                    task_id=task_id,
                    passed=True,
                    steps_taken=obs.step,
                    total_reward=total_reward,
                    artifacts_created=len(
                        self.blackboard.get_artifacts_for_task(root_task.task_id)
                    ),
                    metadata={
                        "budget_summary": self.budget_monitor.get_summary(),
                        "interaction_log": log,
                    },
                )

            if done:
                logger.info(f"Episode ended without success at step {obs.step}")
                break

            # Step 7: Detect conflicts (navigator stuck)
            signals = self.blackboard.detect_signals()
            new_conflicts = [s for s in signals
                             if s.type == SignalType.CONFLICT and s.task_id == root_task.task_id]
            if new_conflicts:
                conflicts_detected += len(new_conflicts)
                logger.info(f"Conflict detected at step {obs.step}")
                if self.mw_logger:
                    self.mw_logger.log_conflict(step=obs.step, conflict_count=conflicts_detected)

            # Track Critic fires (it resolves conflicts after posting REVIEW)
            all_reviews = self.blackboard.get_artifacts_for_task(root_task.task_id, ArtifactType.REVIEW)
            critic_fire_count = len(all_reviews)

            # --- Strategy Reset (branch-and-merge equivalent) ---
            # After Critic has fired twice and task is still stuck, wipe the plan
            # and action history so Planner generates a completely new approach.
            # This is the web equivalent of branch-and-merge: instead of parallel
            # browser forks (impractical), we do a sequential strategy restart.
            if (
                critic_fire_count >= 2
                and strategy_resets == 0
                and not obs.success
            ):
                logger.info(
                    f"Strategy reset: Critic fired {critic_fire_count}x with no success — "
                    "wiping plan + history so Planner generates a new approach"
                )
                strategy_resets += 1

                # Clear plan so Planner re-runs with failure context
                for artifact in self.blackboard.get_artifacts_for_task(root_task.task_id, ArtifactType.PLAN):
                    # Mark artifact superseded by setting version=0 in metadata
                    artifact.metadata["superseded"] = True
                    self.blackboard.post_artifact(artifact)

                # Store failure summary from latest review for Planner to read
                latest_review = all_reviews[-1]
                root_task.metadata["failure_summary"] = latest_review.content[:400]
                root_task.metadata["action_history"] = []  # Fresh start
                root_task.metadata["strategy_reset"] = True
                self.blackboard.update_task(root_task)

                self.metrics.record_branch(n_branches=1)
                logger.info("Strategy reset applied — Planner will replan on next iteration")
                if self.mw_logger:
                    self.mw_logger.log_strategy_reset(
                        step=obs.step,
                        critic_fire_count=critic_fire_count,
                        failure_summary=root_task.metadata.get("failure_summary", ""),
                    )

        # Episode ended without success
        passed, log = runner.result()
        runner.close()
        duration_s = (datetime.utcnow() - start_time).total_seconds()

        self.bb_logger.log_episode_end(problem_id=task_id, passed=False, code="")
        logger.info(f"Episode {task_id} FAILED. Steps: {obs.step if obs else 0}")
        if self.mw_logger:
            self.mw_logger.log_episode_end(
                passed=False,
                steps_taken=obs.step if obs else 0,
                action_history=root_task.metadata.get("action_history", []),
                conflicts=conflicts_detected,
                strategy_resets=strategy_resets,
            )

        return WebEpisodeResult(
            task_id=task_id,
            passed=False,
            steps_taken=obs.step if obs else 0,
            total_reward=total_reward,
            conflicts_detected=conflicts_detected,
            artifacts_created=len(self.blackboard.get_artifacts_for_task(root_task.task_id)),
            metadata={
                "budget_summary": self.budget_monitor.get_summary(),
                "duration_s": duration_s,
                "stop_reason": "max_iterations" if not done else "env_done",
            },
        )

    async def _run_agents_parallel(self) -> None:
        """Run all agents in parallel for one iteration"""
        await asyncio.gather(
            *[agent.run_loop(max_iterations=1) for agent in self.agents],
            return_exceptions=True,
        )
