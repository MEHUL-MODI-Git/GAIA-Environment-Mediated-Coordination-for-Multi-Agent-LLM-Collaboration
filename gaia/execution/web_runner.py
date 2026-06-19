"""Selenium-based MiniWoB++ environment runner

Wraps the Farama MiniWoB++ Gymnasium environment for use by GAIA agents.
Interface mirrors CodeRunner.run_humaneval_test() for consistency.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DOMElement:
    """A single interactive element in the page DOM"""

    element_id: str
    tag: str          # button, input, select, a, etc.
    text: str         # visible text or label
    ref: str          # MiniWoB++ element reference for clicking
    attrs: Dict[str, str] = field(default_factory=dict)
    bbox: Optional[Dict[str, float]] = None  # {left, top, width, height}


@dataclass
class DOMObservation:
    """Structured observation from one MiniWoB++ step"""

    task_id: str
    instruction: str
    elements: List[DOMElement]
    raw_html: str = ""
    step: int = 0
    success: bool = False
    reward: float = 0.0


@dataclass
class WebAction:
    """An action to perform in the browser"""

    type: str           # CLICK, TYPE, SELECT, PRESS_KEY, SCROLL
    ref: int = 0        # element reference integer (for CLICK, TYPE, SELECT)
    text: str = ""      # text to type (for TYPE)
    key: str = ""       # key name (for PRESS_KEY, e.g. "Enter")
    value: str = ""     # option value (for SELECT)
    direction: str = "down"  # scroll direction
    amount: int = 3     # scroll amount


class WebRunner:
    """Run MiniWoB++ tasks using Farama gymnasium + Selenium

    Usage:
        runner = WebRunner(task_name="click-button")
        obs = await runner.reset()
        action = WebAction(type="CLICK", ref=obs.elements[0].ref)
        obs, done = await runner.step(action)
        passed, log = runner.result()
    """

    def __init__(
        self,
        task_name: str,
        max_steps: int = 15,
        headless: bool = True,
        timeout: int = 30,
    ):
        """
        Args:
            task_name: MiniWoB++ task name, e.g. "click-button"
            max_steps: Max browser interaction steps before episode ends
            headless: Run browser in headless mode (no window)
            timeout: Seconds before timing out an episode
        """
        self.task_name = task_name
        self.max_steps = max_steps
        self.headless = headless
        self.timeout = timeout

        self._env = None
        self._step_count = 0
        self._interaction_log: List[str] = []
        self._last_obs: Optional[DOMObservation] = None

    def _get_env(self):
        """Lazily initialize Farama MiniWoB++ environment"""
        if self._env is not None:
            return self._env

        try:
            import miniwob
            import gymnasium as gym

            env_id = f"miniwob/{self.task_name}-v1"
            self._env = gym.make(
                env_id,
                render_mode=None if self.headless else "human",
            )
            return self._env
        except ImportError as e:
            raise ImportError(
                f"MiniWoB++ environment not available: {e}\n"
                "Install with: pip install miniwob gymnasium"
            ) from e

    def _parse_observation(self, raw_obs: Dict, instruction: str) -> DOMObservation:
        """Convert raw gymnasium observation to structured DOMObservation"""
        elements = []

        # MiniWoB++ obs has 'dom_elements' list
        dom_elements = raw_obs.get("dom_elements", [])

        # Pre-pass: build label map for inputs.
        # Two patterns:
        # 1. input_checkbox/input_radio FOLLOWED BY 't' text node — checkbox label after input
        # 2. 'label' element FOLLOWED BY input_* — form field label before input (login etc.)
        label_for_ref: Dict[int, str] = {}
        paired_label_indices: set = set()  # indices of label elements paired with inputs
        for idx, el in enumerate(dom_elements):
            tag = el.get("tag", "")
            # Pattern 1: input_* followed by 't' text label
            if tag.startswith("input_") and el.get("ref", 0) > 0:
                if idx + 1 < len(dom_elements):
                    nxt = dom_elements[idx + 1]
                    if nxt.get("tag") == "t" and nxt.get("text", "").strip():
                        label_for_ref[int(el.get("ref"))] = nxt.get("text", "").strip()
            # Pattern 2: 'label' element followed by input_* — attach label text to input
            elif tag == "label" and el.get("text", "").strip():
                label_text = el.get("text", "").strip()
                for nxt in dom_elements[idx + 1: idx + 4]:  # look ahead up to 3 elements
                    nxt_tag = nxt.get("tag", "")
                    nxt_ref = nxt.get("ref", 0)
                    if nxt_tag.startswith("input_") and nxt_ref and int(str(nxt_ref).lstrip("-") or 0) > 0:
                        label_for_ref[int(nxt_ref)] = label_text
                        paired_label_indices.add(idx)  # this label is paired; can be skipped
                        break

        for i, el in enumerate(dom_elements):
            tag = el.get("tag", "")
            text = el.get("text", "").strip()
            # For input elements, 'value' holds the current typed content
            # (text stays empty even after typing). Show value so agents know what's there.
            value = str(el.get("value", "") or "").strip()
            if not text and value and tag.startswith("input"):
                text = f"[value: {value}]"
            # For checkbox/radio, attach the option label so agents can match by name
            ref_int = int(el.get("ref", i)) if str(el.get("ref", "")).lstrip("-").isdigit() else i
            if not text and ref_int in label_for_ref:
                text = label_for_ref[ref_int]
            # Skip 't' text nodes (content already attached to inputs above)
            # Skip 'label' elements only when they were paired with an input (to avoid duplicates)
            if tag == "t":
                continue
            if tag == "label" and i in paired_label_indices:
                continue
            ref = el.get("ref", str(i))

            # MiniWoB++ uses non-standard tag names: input_text, input_checkbox, input_date etc.
            _INTERACTIVE = {"button", "input", "input_text", "input_checkbox",
                            "input_radio", "input_number", "input_date", "select", "a",
                            "textarea", "option", "t"}
            if tag in _INTERACTIVE or text:
                bbox = None
                if "left" in el:
                    bbox = {
                        "left": el.get("left", 0),
                        "top": el.get("top", 0),
                        "width": el.get("width", 0),
                        "height": el.get("height", 0),
                    }
                elements.append(
                    DOMElement(
                        element_id=f"el_{i}",
                        tag=tag,
                        text=text,
                        ref=ref,
                        attrs={k: str(v) for k, v in el.items()
                               if k not in ("left", "top", "width", "height", "ref", "tag", "text")},
                        bbox=bbox,
                    )
                )

        return DOMObservation(
            task_id=self.task_name,
            instruction=instruction,
            elements=elements,
            raw_html=raw_obs.get("raw_html", ""),
            step=self._step_count,
            success=False,
            reward=0.0,
        )

    async def reset(self) -> DOMObservation:
        """Reset environment and return initial observation"""
        env = self._get_env()
        self._step_count = 0
        self._interaction_log = []

        raw_obs, info = await asyncio.to_thread(env.reset)
        instruction = raw_obs.get("utterance", "")

        self._last_obs = self._parse_observation(raw_obs, instruction)
        self._interaction_log.append(f"RESET: {instruction}")
        return self._last_obs

    async def step(self, action: WebAction) -> Tuple[DOMObservation, bool]:
        """Execute one action and return (observation, done)"""
        env = self._get_env()
        self._step_count += 1

        # Build gymnasium action dict
        gym_action = self._encode_action(action)
        self._interaction_log.append(
            f"STEP {self._step_count}: {action.type} ref={action.ref} text={action.text!r}"
        )

        raw_obs, reward, terminated, truncated, info = await asyncio.to_thread(
            env.step, gym_action
        )
        done = terminated or truncated or self._step_count >= self.max_steps

        instruction = (self._last_obs.instruction if self._last_obs
                       else raw_obs.get("utterance", ""))
        obs = self._parse_observation(raw_obs, instruction)
        obs.success = reward > 0.0
        obs.reward = reward
        obs.step = self._step_count

        self._last_obs = obs
        self._interaction_log.append(
            f"  → reward={reward:.2f} success={obs.success} done={done}"
        )
        return obs, done

    # Mapping from human key names to DEFAULT_ALLOWED_KEYS indices
    _KEY_INDEX = {
        "Enter": 0, "<Enter>": 0, "Return": 0,
        "PageUp": 1, "<PageUp>": 1,
        "PageDown": 2, "<PageDown>": 2,
        "Backspace": 3, "<Backspace>": 3,
        "Delete": 4, "<Delete>": 4,
        "Tab": 5, "<Tab>": 5,
        "Space": 6, "<Space>": 6, " ": 6,
        "ArrowUp": 7, "<ArrowUp>": 7,
        "ArrowRight": 8, "<ArrowRight>": 8,
        "ArrowDown": 9, "<ArrowDown>": 9,
        "ArrowLeft": 10, "<ArrowLeft>": 10,
    }

    def _encode_action(self, action: WebAction) -> Dict[str, Any]:
        """Encode WebAction to gymnasium action dict.

        ActionTypes:
          0  NONE
          8  CLICK_ELEMENT        (ref=element ref int)
          9  PRESS_KEY            (key=index into DEFAULT_ALLOWED_KEYS)
         10  TYPE_TEXT            (text=string)
         12  FOCUS_ELEMENT_AND_TYPE_TEXT  (ref + text)
        """
        action_type_map = {
            "NONE":      0,
            "CLICK":     8,   # CLICK_ELEMENT
            "SELECT":    8,   # CLICK_ELEMENT (clicks on the option/select)
            "PRESS_KEY": 9,   # PRESS_KEY
            "TYPE":      12,  # FOCUS_ELEMENT_AND_TYPE_TEXT
            "SCROLL":    6,   # SCROLL_UP_COORDS (overridden by direction below)
        }
        if action.type == "SCROLL":
            act_int = 6 if action.direction == "up" else 7
        else:
            act_int = action_type_map.get(action.type, 0)

        # Convert key name to index
        key_idx = self._KEY_INDEX.get(action.key, 0)

        return {
            "action_type": act_int,
            "ref": int(action.ref) if action.ref else 0,
            "text": action.text or action.value,  # value used for SELECT text
            "key": key_idx,
            "coords": [0.0, 0.0],
            "field": 0,
        }

    def result(self) -> Tuple[bool, str]:
        """Return (passed, interaction_log) after episode ends"""
        passed = self._last_obs.success if self._last_obs else False
        log = "\n".join(self._interaction_log)
        return passed, log

    def close(self):
        """Close the browser environment"""
        if self._env is not None:
            self._env.close()
            self._env = None

    async def run_task(
        self,
        action_sequence: List[Dict[str, str]],
    ) -> Tuple[bool, str]:
        """Run a complete action sequence and return (passed, log).

        Mirrors CodeRunner.run_humaneval_test() interface.

        Args:
            action_sequence: List of action dicts from agent
                e.g. [{"type": "CLICK", "ref": "123"}, {"type": "TYPE", "text": "hello"}]

        Returns:
            Tuple of (passed: bool, interaction_log: str)
        """
        try:
            await self.reset()

            for action_dict in action_sequence:
                action = WebAction(
                    type=action_dict.get("type", "NONE"),
                    ref=action_dict.get("ref", ""),
                    text=action_dict.get("text", ""),
                    key=action_dict.get("key", ""),
                    value=action_dict.get("value", ""),
                )
                obs, done = await self.step(action)
                if obs.success:
                    return True, "\n".join(self._interaction_log)
                if done:
                    break

            return self.result()

        except Exception as e:
            return False, f"Environment error: {e}"
        finally:
            self.close()
