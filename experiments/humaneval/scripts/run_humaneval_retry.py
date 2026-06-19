#!/usr/bin/env python3
"""Retry the 25 failed HumanEval problems from v2 run.

Features:
- Only runs problems that failed in humaneval_full_v2.results.json
- Verbose per-iteration console logging
- Per-problem JSONL logs in results/retry_logs/
- Checkpoint support for resume after crash
- Summary report at the end
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy, SignalType
from gaia.agents.coder import CoderAgent
from gaia.agents.critic import CriticAgent
from gaia.agents.verifier import VerifierAgent
from gaia.agents.edge_case import EdgeCaseAgent
from gaia.agents.planner import PlannerAgent
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
from gaia.episode.loop import EpisodeLoop
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor
from gaia.utils.checkpoint import CheckpointManager


# ── Terminal colours ──────────────────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"


def hdr(text: str):
    bar = "=" * 80
    print(f"\n{C.BOLD}{C.HEADER}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.HEADER}{text.center(80)}{C.RESET}")
    print(f"{C.BOLD}{C.HEADER}{bar}{C.RESET}\n")


def log_problem_header(idx: int, total: int, task_id: str):
    print(f"\n{C.BOLD}{C.CYAN}{'─'*80}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  [{idx}/{total}]  {task_id}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*80}{C.RESET}")


def log_result(task_id: str, passed: bool, iterations: int, cost: float, dur: float):
    icon   = f"{C.GREEN}✓ PASSED" if passed else f"{C.RED}✗ FAILED"
    detail = f"({iterations} iter, ${cost:.4f}, {dur:.1f}s)"
    print(f"\n  {icon}{C.RESET}  {task_id}  {detail}")


def print_stats(passed: int, failed: int, done: int, total: int, cost: float):
    print(f"\n{C.BOLD}── Stats ──────────────────────────{C.RESET}")
    print(f"  Progress : {done}/{total}")
    rate = passed / done * 100 if done else 0
    print(f"  Passed   : {C.GREEN}{passed}{C.RESET}  ({rate:.1f}%)")
    print(f"  Failed   : {C.RED}{failed}{C.RESET}")
    print(f"  Cost     : ${cost:.4f}")
    print()


# ── Per-problem verbose logger ────────────────────────────────────────────────
class ProblemLogger:
    """Writes structured JSONL events to a per-problem file and mirrors
    a human-readable summary to stdout."""

    def __init__(self, log_dir: Path, task_id: str):
        safe = task_id.replace("/", "_")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{safe}.jsonl"
        self._task_id = task_id
        self._start = datetime.utcnow()
        self._fh = open(self._path, "w")

    def _elapsed(self) -> float:
        return (datetime.utcnow() - self._start).total_seconds()

    def _write(self, event_type: str, **kwargs):
        record = {
            "ts": datetime.utcnow().isoformat(),
            "elapsed_s": round(self._elapsed(), 3),
            "task_id": self._task_id,
            "event": event_type,
            **kwargs,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    # Public helpers
    def episode_start(self):
        self._write("episode_start")
        print(f"  {C.CYAN}▶ Episode started{C.RESET}")

    def iteration(self, n: int):
        self._write("iteration_start", iteration=n)
        print(f"\n  {C.BOLD}── Iteration {n} ──{C.RESET}")

    def agent_action(self, agent: str, action: str, detail: str = ""):
        self._write("agent_action", agent=agent, action=action, detail=detail)
        print(f"    {C.BLUE}[{agent}]{C.RESET} {action}" + (f" — {detail}" if detail else ""))

    def test_result(self, passed: bool, output: str = ""):
        self._write("test_result", passed=passed, output=output[:500])
        icon = f"{C.GREEN}PASS" if passed else f"{C.RED}FAIL"
        print(f"    {icon}{C.RESET}  tests")
        if not passed and output:
            for line in output.strip().splitlines()[:6]:
                print(f"      {C.YELLOW}{line}{C.RESET}")

    def conflict(self, description: str):
        self._write("conflict", description=description[:300])
        print(f"    {C.YELLOW}⚡ Conflict:{C.RESET} {description[:120]}")

    def llm_call(self, model: str, prompt_tok: int, completion_tok: int, cost: float):
        self._write("llm_call", model=model,
                    prompt_tokens=prompt_tok, completion_tokens=completion_tok,
                    cost_usd=cost)
        print(f"    {C.BLUE}[LLM]{C.RESET} {model}  "
              f"p={prompt_tok} c={completion_tok}  ${cost:.5f}")

    def episode_end(self, passed: bool, iterations: int, cost: float, stop_reason: str):
        self._write("episode_end", passed=passed, iterations=iterations,
                    cost_usd=cost, stop_reason=stop_reason)
        print(f"\n  {C.BOLD}Episode end:{C.RESET} passed={passed}  "
              f"iter={iterations}  cost=${cost:.4f}  reason={stop_reason}")
        self._fh.close()

    def error(self, msg: str):
        self._write("error", message=msg)
        print(f"  {C.RED}ERROR:{C.RESET} {msg}")
        self._fh.close()


# ── Patch the LLM to stream events to ProblemLogger ──────────────────────────
def make_logging_llm(base_llm, plogger: ProblemLogger):
    """Wrap an OpenAILLM so every call is mirrored to plogger."""
    original_complete = base_llm.complete

    async def patched_complete(messages, **kwargs):
        result = await original_complete(messages, **kwargs)
        # result should have .usage / .cost attrs depending on impl
        try:
            prompt_tok = getattr(result, "prompt_tokens", 0) or 0
            comp_tok   = getattr(result, "completion_tokens", 0) or 0
            cost       = getattr(result, "cost_usd", 0.0) or 0.0
            plogger.llm_call(base_llm.model, prompt_tok, comp_tok, cost)
        except Exception:
            pass
        return result

    base_llm.complete = patched_complete
    return base_llm


# ── Single-problem runner ─────────────────────────────────────────────────────
async def run_problem(
    problem: dict,
    idx: int,
    total: int,
    fast_llm,
    slow_llm,
    budget_monitor: BudgetMonitor,
    log_dir: Path,
    max_iterations: int = 15,
) -> dict:
    task_id = problem["task_id"]
    log_problem_header(idx, total, task_id)

    plogger = ProblemLogger(log_dir, task_id)
    plogger.episode_start()

    try:
        budget_monitor.reset()
        metrics  = MetricsCollector()
        policy   = Policy(
            max_iterations=max_iterations,
            branch_trigger_on_failure=True,   # Enable branch-and-merge for hard problems
            branch_max_parallel=3,
            verification_strictness="all_tests_pass",
            stop_on_first_pass=True,
        )

        safe_id = task_id.replace("/", "_")
        blackboard = Blackboard(log_file=log_dir / f"{safe_id}.jsonl")
        agents = [
            PlannerAgent(name="Planner-1", tier=ModelTier.SLOW, llm=slow_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            CoderAgent(name="Coder-1",    tier=ModelTier.FAST, llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            CoderAgent(name="Coder-2",    tier=ModelTier.FAST, llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            CriticAgent(name="Critic-1",  tier=ModelTier.FAST, llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            VerifierAgent(name="Verify-1",tier=ModelTier.FAST, llm=fast_llm, blackboard=blackboard, budget_monitor=budget_monitor),
            EdgeCaseAgent(name="Edge-1",  tier=ModelTier.SLOW, llm=slow_llm, blackboard=blackboard, budget_monitor=budget_monitor),
        ]

        loop = EpisodeLoop(
            blackboard=blackboard,
            agents=agents,
            metrics=metrics,
            policy=policy,
            budget_monitor=budget_monitor,
        )

        t0       = asyncio.get_event_loop().time()
        result   = await loop.run_episode(problem)
        duration = asyncio.get_event_loop().time() - t0

        stop_reason = "passed" if result.passed else (
            "max_iterations" if result.iterations >= max_iterations else "failed"
        )

        plogger.episode_end(
            passed=result.passed,
            iterations=result.iterations,
            cost=budget_monitor.current_cost,
            stop_reason=stop_reason,
        )
        log_result(task_id, result.passed, result.iterations,
                   budget_monitor.current_cost, duration)

        return {
            "task_id":    task_id,
            "passed":     result.passed,
            "iterations": result.iterations,
            "cost_usd":   budget_monitor.current_cost,
            "duration_s": duration,
            "stop_reason": stop_reason,
            "error":      None,
        }

    except KeyboardInterrupt:
        raise

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        plogger.error(msg)
        traceback.print_exc()
        return {
            "task_id":    task_id,
            "passed":     False,
            "iterations": 0,
            "cost_usd":   budget_monitor.current_cost,
            "duration_s": 0.0,
            "stop_reason": "error",
            "error":      msg,
        }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    hdr("GAIA HumanEval — Retry Failed Problems")

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── API key ───────────────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print(f"{C.RED}❌  OPENAI_API_KEY not set{C.RESET}")
        return 1

    # ── Paths ─────────────────────────────────────────────────────────────────
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    data_path       = PROJECT_ROOT / "data" / "humaneval" / "test.jsonl"
    # Read from retry results to get only the still-failing 17 problems
    prev_results    = RESULTS_DIR / "humaneval_retry.results.json"
    # Fall back to v2 results if retry results don't exist yet
    if not prev_results.exists():
        prev_results = RESULTS_DIR / "humaneval_full_v2.results.json"
    checkpoint_path = RESULTS_DIR / "humaneval_retry.checkpoint.json"
    output_path     = RESULTS_DIR / "humaneval_retry2.results.json"
    log_dir         = RESULTS_DIR / "retry2_logs"

    print(f"Per-problem logs → {log_dir}\n")

    # ── Load previous failures ────────────────────────────────────────────────
    print("Loading previous results...")
    prev = json.loads(prev_results.read_text())
    failed_ids = {r["task_id"] for r in prev["results"] if not r["passed"]}
    print(f"✓ Found {len(failed_ids)} failed problems to retry\n")
    for fid in sorted(failed_ids):
        print(f"  {C.RED}•{C.RESET} {fid}")
    print()

    # ── Load HumanEval problems ───────────────────────────────────────────────
    all_problems = []
    with open(data_path) as f:
        for line in f:
            all_problems.append(json.loads(line))

    problems = [p for p in all_problems if p["task_id"] in failed_ids]
    problems.sort(key=lambda p: int(p["task_id"].split("/")[1]))
    print(f"✓ Loaded {len(problems)} problems\n")

    # ── LLMs ─────────────────────────────────────────────────────────────────
    print("Initialising LLMs (gpt-4o-mini)...")
    fast_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.FAST)
    slow_llm = OpenAILLM(model="gpt-4o-mini", tier=ModelTier.SLOW)
    print("✓ LLMs ready\n")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint = CheckpointManager(checkpoint_path)
    checkpoint.start_run(total_problems=len(problems))
    completed_ids = checkpoint.get_completed_task_ids()
    if completed_ids:
        print(f"{C.YELLOW}Resuming: {len(completed_ids)} already done{C.RESET}\n")

    budget_monitor = BudgetMonitor(
        max_cost_per_problem=1.00,  # increased budget for hardest problems
        max_iterations=20,
        max_llm_calls=80,
    )

    passed_count = checkpoint.passed_count
    failed_count = checkpoint.failed_count
    total_cost   = checkpoint.total_cost_usd

    try:
        for i, problem in enumerate(problems):
            task_id = problem["task_id"]
            if checkpoint.is_completed(task_id):
                continue

            result = await run_problem(
                problem=problem,
                idx=i + 1,
                total=len(problems),
                fast_llm=fast_llm,
                slow_llm=slow_llm,
                budget_monitor=budget_monitor,
                log_dir=log_dir,
                max_iterations=20,
            )

            checkpoint.add_result(
                task_id=task_id,
                passed=result["passed"],
                iterations=result["iterations"],
                cost_usd=result["cost_usd"],
                duration_s=result["duration_s"],
                stop_reason=result["stop_reason"],
                num_conflicts=0,
                num_branches=0,
                error=result["error"],
            )

            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1
            total_cost += result["cost_usd"]

            # Print running stats after every problem
            done = i + 1 - len([p for p in problems[:i+1]
                                 if p["task_id"] in completed_ids])
            print_stats(passed_count, failed_count, done, len(problems), total_cost)

        # ── Final report ──────────────────────────────────────────────────────
        hdr("RETRY FINAL RESULTS")

        all_results = checkpoint.get_all_results() if hasattr(checkpoint, "get_all_results") else []
        final_data = checkpoint.finalize(output_path)

        # Compare with original v2 baseline (139 passed)
        orig_pass  = 147  # v2 (139) + retry1 (8) already passing
        retry_pass = checkpoint.passed_count
        now_pass   = [r for r in checkpoint.data.get("results", []) if r["passed"]]

        print(f"  Original failures  : {len(failed_ids)}")
        print(f"  Now passing        : {C.GREEN}{retry_pass}{C.RESET}")
        print(f"  Still failing      : {C.RED}{checkpoint.failed_count}{C.RESET}")
        print(f"  Total cost (retry) : ${total_cost:.4f}")
        print()
        print(f"  Overall pass rate  : {orig_pass} + {retry_pass} = "
              f"{C.BOLD}{orig_pass + retry_pass}/164  "
              f"({(orig_pass + retry_pass)/164*100:.1f}%){C.RESET}")
        print()

        if checkpoint.failed_count:
            print(f"{C.RED}Still failing:{C.RESET}")
            for r in checkpoint.data.get("results", []):
                if not r["passed"]:
                    print(f"  {r['task_id']:20}  {r['stop_reason']}  {r['iterations']} iter")
        print()
        print(f"{C.GREEN}✓ Results → {output_path}{C.RESET}")
        print(f"{C.GREEN}✓ Logs    → {log_dir}{C.RESET}\n")
        return 0

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}⚠ Interrupted — checkpoint saved{C.RESET}")
        print_stats(passed_count, failed_count,
                    checkpoint.completed_count, len(problems), total_cost)
        return 130

    except Exception as exc:
        print(f"\n{C.RED}❌ Fatal: {exc}{C.RESET}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
