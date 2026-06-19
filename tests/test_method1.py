"""Quick test of Method 1 on first few HumanEval problems"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaia.benchmarks.humaneval.loader import HumanEvalLoader
from gaia.methods.single_agent import SingleAgentMethod
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.base import ModelTier
import os


async def main():
    """Test Method 1 on first 3 problems"""
    print("\n" + "="*60)
    print("Testing Method 1: Single Agent Baseline")
    print("="*60 + "\n")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it:")
        print("  export OPENAI_API_KEY='sk-...'")
        return

    print("✓ OpenAI API key found\n")

    # Load first 3 problems
    loader = HumanEvalLoader(Path("data/humaneval/test.jsonl"))
    problems = loader.load_range(0, 3)
    print(f"✓ Loaded {len(problems)} test problems\n")

    # Initialize method
    llm = OpenAILLM(
        model="gpt-4o-mini",
        tier=ModelTier.FAST,
        api_key=api_key,
        temperature=0.0,  # Deterministic for testing
    )
    method = SingleAgentMethod(llm=llm, max_retries=2)
    print("✓ Initialized SingleAgentMethod with gpt-4o-mini\n")

    # Run tests
    print("Running tests...\n")
    results = []

    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {problem['task_id']}: {problem['entry_point']}()")

        try:
            result = await method.solve(problem)

            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} - {result.iterations} iterations, "
                  f"${result.cost_usd:.4f}, {result.latency_ms:.0f}ms")

            results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")

        print()

    # Summary
    print("="*60)
    print("Summary")
    print("="*60)

    passed = sum(1 for r in results if r.passed)
    total_cost = sum(r.cost_usd for r in results)
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

    print(f"Passed: {passed}/{len(results)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    print()

    if passed == len(results):
        print("🎉 All tests passed! Method 1 is working correctly.")
    elif passed > 0:
        print("⚠️  Some tests passed. Method 1 is partially working.")
    else:
        print("❌ No tests passed. Check the implementation.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
