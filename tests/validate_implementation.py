"""Quick validation script to test GAIA implementation"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules import correctly"""
    print("Testing imports...")

    # Core modules
    from gaia.blackboard.models import Task, Artifact, Policy, Signal, Evidence
    from gaia.blackboard.storage import InMemoryStorage
    from gaia.blackboard.blackboard import Blackboard
    print("✓ Blackboard modules")

    # LLM providers
    from gaia.llms.base import BaseLLM, ModelTier, LLMResult
    from gaia.llms.openai_llm import OpenAILLM
    from gaia.llms.anthropic_llm import AnthropicLLM
    from gaia.llms.groq_llm import GroqLLM
    from gaia.llms.gemini_llm import GeminiLLM
    print("✓ LLM modules")

    # Agents
    from gaia.agents.base import BaseAgent
    from gaia.agents.coder import CoderAgent
    from gaia.agents.critic import CriticAgent
    from gaia.agents.verifier import VerifierAgent
    from gaia.agents.planner import PlannerAgent
    print("✓ Agent modules")

    # Episode coordination
    from gaia.episode.loop import EpisodeLoop, EpisodeResult
    from gaia.episode.scheduler import Scheduler
    print("✓ Episode modules")

    # Resolution
    from gaia.resolution.conflict import ConflictDetector
    from gaia.resolution.branch_merge import BranchManager
    print("✓ Resolution modules")

    # Meta
    from gaia.meta.policy import PolicyManager
    from gaia.meta.meta_update import MetaUpdater
    print("✓ Meta modules")

    # Methods
    from gaia.methods.base import BaseMethod, MethodResult
    from gaia.methods.single_agent import SingleAgentMethod
    from gaia.methods.multi_agent_chat import MultiAgentChatMethod
    from gaia.methods.gaia_ae import GAIAAEMethod
    from gaia.methods.gaia_af import GAIAAFMethod
    from gaia.methods.gaia_ag import GAIAAGMethod
    print("✓ Method modules")

    # Utilities
    from gaia.utils.registry import Registry
    from gaia.utils.logging import get_logger
    from gaia.utils.metrics import MetricsCollector
    print("✓ Utility modules")

    # Benchmarks
    from gaia.benchmarks.humaneval.loader import HumanEvalLoader
    from gaia.benchmarks.humaneval.evaluator import HumanEvalEvaluator
    print("✓ Benchmark modules")

    # Execution
    from gaia.execution.code_runner import CodeRunner
    from gaia.execution.verifier import VerificationGate
    print("✓ Execution modules")

    # Parsers
    from gaia.parsers.code_parser import CodeParser
    from gaia.parsers.plan_parser import PlanParser
    from gaia.parsers.review_parser import ReviewParser
    print("✓ Parser modules")

    print("\n✓ All imports successful!")
    return True


def test_basic_objects():
    """Test that basic objects can be created"""
    print("\nTesting object creation...")

    from gaia.blackboard.models import Task, Policy
    from gaia.blackboard.storage import InMemoryStorage
    from gaia.blackboard.blackboard import Blackboard

    # Create policy
    policy = Policy(max_iterations=5)
    print(f"✓ Policy created: max_iterations={policy.max_iterations}")

    # Create storage
    storage = InMemoryStorage()
    print("✓ Storage created")

    # Create blackboard
    blackboard = Blackboard(storage=storage, policy=policy)
    print("✓ Blackboard created")

    # Create and post task
    task = Task(
        title="Test task",
        description="Testing blackboard",
        acceptance_criteria="Should work",
    )
    blackboard.post_task(task)
    print(f"✓ Task posted: {task.task_id}")

    # Retrieve task
    retrieved = blackboard.get_task(task.task_id)
    assert retrieved is not None
    assert retrieved.title == "Test task"
    print("✓ Task retrieved successfully")

    return True


def test_registry():
    """Test registry pattern"""
    print("\nTesting registry pattern...")

    from gaia.utils.registry import Registry

    test_registry = Registry(name="TestRegistry")

    @test_registry.register("test_class")
    class TestClass:
        def __init__(self, value: int):
            self.value = value

    # Build instance
    instance = test_registry.build("test_class", value=42)
    assert instance.value == 42
    print("✓ Registry pattern working")

    return True


def main():
    """Run all validation tests"""
    print("="*60)
    print("GAIA Implementation Validation")
    print("="*60)

    try:
        # Run tests
        test_imports()
        test_basic_objects()
        test_registry()

        print("\n" + "="*60)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("="*60)
        print("\nGAIA framework is ready to use!")
        print("\nNext steps:")
        print("1. Set your API keys (OPENAI_API_KEY, etc.)")
        print("2. Run: python scripts/run_experiment.py --method single_agent \\")
        print("           --data data/humaneval/test.jsonl \\")
        print("           --output results/test.jsonl --problems 0-2")
        print()
        return 0

    except Exception as e:
        print(f"\n✗ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
