#!/usr/bin/env python3
"""Quick test script to verify GAIA framework on a few HumanEval problems

Tests the complete framework with logging and budget monitoring.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaia.blackboard.blackboard import Blackboard
from gaia.blackboard.models import Policy
from gaia.agents.coder import CoderAgent
from gaia.agents.critic import CriticAgent
from gaia.agents.verifier import VerifierAgent
from gaia.agents.edge_case import EdgeCaseAgent
from gaia.llms.base import ModelTier
from gaia.llms.openai_llm import OpenAILLM
from gaia.episode.loop import EpisodeLoop
from gaia.utils.metrics import MetricsCollector
from gaia.utils.budget_monitor import BudgetMonitor


async def test_framework():
    """Test framework on first 3 HumanEval problems"""

    print("=" * 80)
    print("GAIA Framework Test")
    print("=" * 80)

    # Load test problems
    data_path = Path(__file__).parent.parent / "data" / "humaneval" / "test.jsonl"
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        print("Please ensure HumanEval data is available.")
        return

    problems = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i == 91:  # Load HumanEval/91
                problems.append(json.loads(line))
                break

    print(f"\n✓ Loaded {len(problems)} test problem(s)")

    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set it before running:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    print("✓ OpenAI API key found")

    # Create LLMs
    print("\nInitializing LLMs...")
    fast_llm = OpenAILLM(
        model="gpt-4o-mini",
        tier=ModelTier.FAST,
        temperature=0.7,
        max_tokens=2048
    )
    slow_llm = OpenAILLM(
        model="gpt-4o",
        tier=ModelTier.SLOW,
        temperature=0.7,
        max_tokens=2048
    )
    print("✓ LLMs initialized")

    # Create blackboard with logging
    log_dir = Path(__file__).parent.parent / "results" / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run each problem
    results = []
    for i, problem in enumerate(problems):
        print("\n" + "=" * 80)
        print(f"Problem {i+1}/{len(problems)}: {problem['task_id']}")
        print("=" * 80)

        # Create fresh blackboard for each problem
        blackboard = Blackboard(
            log_file=log_dir / f"problem_{i:03d}.jsonl"
        )

        # Create budget monitor ($0.30 per problem)
        budget_monitor = BudgetMonitor(
            max_cost_per_problem=0.30,
            max_iterations=10,
            max_llm_calls=30,
            warn_threshold=0.75
        )

        # Create agents (no Planner for HumanEval - tasks are too simple)
        agents = [
            CoderAgent(
                name="Coder-1",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                budget_monitor=budget_monitor
            ),
            CoderAgent(
                name="Coder-2",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                budget_monitor=budget_monitor
            ),
            CriticAgent(
                name="Critic-1",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                budget_monitor=budget_monitor
            ),
            VerifierAgent(
                name="Verifier-1",
                tier=ModelTier.FAST,
                llm=fast_llm,
                blackboard=blackboard,
                budget_monitor=budget_monitor
            ),
            EdgeCaseAgent(
                name="EdgeCase-1",
                tier=ModelTier.SLOW,
                llm=slow_llm,
                blackboard=blackboard,
                budget_monitor=budget_monitor
            ),
        ]

        # Create metrics
        metrics = MetricsCollector()

        # Create policy
        policy = Policy(
            max_iterations=10,
            branch_trigger_on_failure=False,  # Disable branching for quick test
            verification_strictness="all_tests_pass",
            stop_on_first_pass=True
        )

        # Create episode loop
        episode = EpisodeLoop(
            blackboard=blackboard,
            agents=agents,
            metrics=metrics,
            policy=policy,
            budget_monitor=budget_monitor
        )

        # Run episode
        try:
            result = await episode.run_episode(problem)

            # Display results
            print(f"\n{'✓' if result.passed else '✗'} Result: {'PASSED' if result.passed else 'FAILED'}")
            print(f"   Iterations: {result.iterations}")

            # Budget summary
            budget_summary = result.metadata.get("budget_summary", {})
            print(f"\n💰 Budget Used:")
            print(f"   Cost: ${budget_summary.get('cost_usd', 0):.4f} / ${budget_summary.get('cost_limit_usd', 0.30):.2f}")
            print(f"   LLM calls: {budget_summary.get('llm_calls', 0)} / {budget_summary.get('llm_call_limit', 30)}")

            # Episode summary
            episode_summary = result.metadata.get("episode_summary", {})
            print(f"\n📊 Episode Summary:")
            print(f"   Total events: {episode_summary.get('total_events', 0)}")
            print(f"   Total tokens: {episode_summary.get('total_tokens', 0):,}")
            print(f"   Artifacts: {episode_summary.get('artifacts_created', 0)}")
            print(f"   Conflicts: {episode_summary.get('conflicts_detected', 0)}")
            print(f"   Duration: {episode_summary.get('duration_seconds', 0):.1f}s")

            # Save result
            results.append({
                "problem_id": problem["task_id"],
                "passed": result.passed,
                "iterations": result.iterations,
                "cost_usd": budget_summary.get('cost_usd', 0),
                "duration_s": episode_summary.get('duration_seconds', 0),
                "stop_reason": result.metadata.get("stop_reason", "success" if result.passed else "unknown")
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "problem_id": problem["task_id"],
                "passed": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for r in results if r.get("passed"))
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    avg_duration = sum(r.get("duration_s", 0) for r in results) / len(results) if results else 0

    print(f"\nResults: {passed_count}/{len(results)} passed ({passed_count/len(results)*100:.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average duration: {avg_duration:.1f}s per problem")

    print(f"\n📁 Detailed logs saved to: {log_dir}")

    # Save summary
    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "test_date": str(Path(__file__).parent.parent),
            "problems_tested": len(results),
            "passed": passed_count,
            "total_cost_usd": total_cost,
            "average_duration_s": avg_duration,
            "results": results
        }, f, indent=2)

    print(f"📄 Summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    try:
        asyncio.run(test_framework())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
