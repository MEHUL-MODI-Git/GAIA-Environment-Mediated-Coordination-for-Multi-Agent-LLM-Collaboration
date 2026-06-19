"""CLI runner for GAIA experiments

Usage:
    python scripts/run_experiment.py \\
        --method single_agent \\
        --data data/humaneval/test.jsonl \\
        --output results/method1/results.jsonl \\
        --problems 0-5
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gaia.benchmarks.humaneval.loader import HumanEvalLoader
from gaia.benchmarks.humaneval.evaluator import HumanEvalEvaluator
from gaia.methods.single_agent import SingleAgentMethod
from gaia.methods.multi_agent_chat import MultiAgentChatMethod
from gaia.methods.gaia_ae import GAIAAEMethod
from gaia.methods.gaia_af import GAIAAFMethod
from gaia.methods.gaia_ag import GAIAAGMethod
from gaia.llms.openai_llm import OpenAILLM
from gaia.llms.anthropic_llm import AnthropicLLM
from gaia.llms.groq_llm import GroqLLM
from gaia.llms.gemini_llm import GeminiLLM
from gaia.llms.base import ModelTier
from gaia.utils.logging import get_logger

logger = get_logger("run_experiment")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run GAIA experiment on HumanEval")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["single_agent", "multi_agent_chat", "gaia_ae", "gaia_af", "gaia_ag"],
        help="Experiment method to run",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HumanEval test.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for results.jsonl",
    )

    parser.add_argument(
        "--problems",
        type=str,
        default="0-164",
        help="Problem range to run (e.g., '0-10' or '5-15')",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "groq", "gemini"],
        help="LLM provider",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to environment variable)",
    )

    return parser.parse_args()


def parse_problem_range(range_str: str) -> tuple:
    """Parse problem range string like '0-10' to (start, end)"""
    parts = range_str.split("-")
    start = int(parts[0])
    end = int(parts[1])
    return start, end


def create_llm(provider: str, model: str, tier: ModelTier, api_key=None):
    """Create LLM instance based on provider"""
    if provider == "openai":
        return OpenAILLM(
            model=model,
            tier=tier,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )
    elif provider == "anthropic":
        return AnthropicLLM(
            model=model,
            tier=tier,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
        )
    elif provider == "groq":
        return GroqLLM(
            model=model,
            tier=tier,
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
        )
    elif provider == "gemini":
        return GeminiLLM(
            model=model,
            tier=tier,
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def run_single_agent(args, problems):
    """Run Method 1: Single Agent Baseline"""
    llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    method = SingleAgentMethod(llm=llm, max_retries=args.max_retries)
    return await run_method(method, args, problems)


async def run_multi_agent_chat(args, problems):
    """Run Method 2: Multi-Agent Chat"""
    solver_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    critic_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)

    method = MultiAgentChatMethod(
        solver_llm=solver_llm,
        critic_llm=critic_llm,
        max_rounds=5,
        max_retries=args.max_retries,
    )
    return await run_method(method, args, problems)


async def run_gaia_ae(args, problems):
    """Run Method 3: GAIA A-E"""
    coder_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    critic_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    # Use a stronger model for verification (or same if specified)
    verifier_model = args.model.replace("-mini", "") if "-mini" in args.model else args.model
    verifier_llm = create_llm(args.provider, verifier_model, ModelTier.SLOW, args.api_key)

    method = GAIAAEMethod(
        coder_llm=coder_llm,
        critic_llm=critic_llm,
        verifier_llm=verifier_llm,
        max_iterations=10,
        max_retries=args.max_retries,
    )
    return await run_method(method, args, problems)


async def run_gaia_af(args, problems):
    """Run Method 4: GAIA A-F"""
    coder_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    critic_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    verifier_model = args.model.replace("-mini", "") if "-mini" in args.model else args.model
    verifier_llm = create_llm(args.provider, verifier_model, ModelTier.SLOW, args.api_key)

    method = GAIAAFMethod(
        coder_llm=coder_llm,
        critic_llm=critic_llm,
        verifier_llm=verifier_llm,
        max_iterations=10,
        max_retries=args.max_retries,
        branch_max_parallel=3,
    )
    return await run_method(method, args, problems)


async def run_gaia_ag(args, problems):
    """Run Method 5: GAIA A-G"""
    coder_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    critic_llm = create_llm(args.provider, args.model, ModelTier.FAST, args.api_key)
    verifier_model = args.model.replace("-mini", "") if "-mini" in args.model else args.model
    verifier_llm = create_llm(args.provider, verifier_model, ModelTier.SLOW, args.api_key)

    method = GAIAAGMethod(
        coder_llm=coder_llm,
        critic_llm=critic_llm,
        verifier_llm=verifier_llm,
        max_iterations=10,
        max_retries=args.max_retries,
        branch_max_parallel=3,
        meta_update_frequency=10,
    )
    return await run_method(method, args, problems)


async def run_method(method, args, problems):
    """Run a method on problems and collect results"""
    results = []
    for problem in tqdm(problems, desc=f"Running {args.method}"):
        try:
            result = await method.solve(problem)
            results.append(result.model_dump())

            # Write incrementally
            append_jsonl(args.output, result.model_dump())

            # Log progress
            status = "✓" if result.passed else "✗"
            logger.info(
                f"{status} {problem['task_id']} - "
                f"Iterations: {result.iterations}, "
                f"Cost: ${result.total_cost_usd:.4f}"
            )

        except Exception as e:
            logger.error(f"Error on {problem['task_id']}: {str(e)}", exc_info=True)
            error_result = {
                "task_id": problem["task_id"],
                "method": args.method,
                "passed": False,
                "error": str(e),
            }
            results.append(error_result)
            append_jsonl(args.output, error_result)

    return results


def append_jsonl(path: str, data: dict):
    """Append JSON line to file"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


async def main():
    """Main entry point"""
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"GAIA Experiment Runner")
    print(f"{'='*60}")
    print(f"Method: {args.method}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Problem range: {args.problems}")
    print(f"{'='*60}\n")

    # Load problems
    loader = HumanEvalLoader(Path(args.data))
    start, end = parse_problem_range(args.problems)
    problems = loader.load_range(start, end)

    print(f"Loaded {len(problems)} problems\n")

    # Run experiment
    start_time = datetime.now()

    if args.method == "single_agent":
        results = await run_single_agent(args, problems)
    elif args.method == "multi_agent_chat":
        results = await run_multi_agent_chat(args, problems)
    elif args.method == "gaia_ae":
        results = await run_gaia_ae(args, problems)
    elif args.method == "gaia_af":
        results = await run_gaia_af(args, problems)
    elif args.method == "gaia_ag":
        results = await run_gaia_ag(args, problems)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Compute metrics
    evaluator = HumanEvalEvaluator()
    metrics = evaluator.compute_metrics_summary(results)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total problems: {metrics['total_problems']}")
    print(f"Pass@1: {metrics['pass_at_1']:.1%}")
    print(f"Total cost: ${metrics['total_cost_usd']:.2f}")
    print(f"Avg latency: {metrics['avg_latency_ms']:.0f}ms")
    print(f"Avg iterations: {metrics['avg_iterations']:.1f}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
