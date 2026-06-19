#!/usr/bin/env python3
"""Test checkpoint system on first 5 problems"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gaia.utils.checkpoint import CheckpointManager


async def main():
    checkpoint_path = PROJECT_ROOT / "results" / "test_checkpoint.json"
    output_path = PROJECT_ROOT / "results" / "test_checkpoint_final.json"

    # Clean up any existing checkpoint
    checkpoint_path.unlink(missing_ok=True)

    # Test 1: Create new checkpoint
    print("Test 1: Creating new checkpoint...")
    cp = CheckpointManager(checkpoint_path)
    cp.start_run(total_problems=5)
    print(f"✓ Checkpoint created: {cp.completed_count}/{cp.total_problems}")

    # Test 2: Add some results
    print("\nTest 2: Adding results...")
    cp.add_result(
        task_id="HumanEval/0",
        passed=True,
        iterations=3,
        cost_usd=0.0005,
        duration_s=10.5,
        stop_reason="passed",
    )
    cp.add_result(
        task_id="HumanEval/1",
        passed=False,
        iterations=10,
        cost_usd=0.0020,
        duration_s=45.2,
        stop_reason="max_iterations",
    )
    print(f"✓ Added 2 results: {cp.completed_count}/{cp.total_problems}")
    print(f"  Pass rate: {cp.pass_rate * 100:.1f}%")

    # Test 3: Reload checkpoint
    print("\nTest 3: Reloading checkpoint...")
    cp2 = CheckpointManager(checkpoint_path)
    print(f"✓ Reloaded: {cp2.completed_count}/{cp2.total_problems}")
    print(f"  Pass rate: {cp2.pass_rate * 100:.1f}%")
    print(f"  Completed IDs: {cp2.get_completed_task_ids()}")

    # Test 4: Check is_completed
    print("\nTest 4: Checking is_completed...")
    print(f"  HumanEval/0 completed? {cp2.is_completed('HumanEval/0')}")
    print(f"  HumanEval/2 completed? {cp2.is_completed('HumanEval/2')}")

    # Test 5: Add error case
    print("\nTest 5: Adding error case...")
    cp2.add_result(
        task_id="HumanEval/2",
        passed=False,
        iterations=0,
        cost_usd=0.0001,
        duration_s=0.5,
        stop_reason="error",
        error="NetworkError: Connection timeout",
    )
    print(f"✓ Added error: {cp2.error_count} errors total")

    # Test 6: Finalize
    print("\nTest 6: Finalizing...")
    final_data = cp2.finalize(output_path)
    print(f"✓ Finalized: {final_data['passed']}/{final_data['total_problems']} passed")
    print(f"  Checkpoint deleted? {not checkpoint_path.exists()}")
    print(f"  Final results saved? {output_path.exists()}")

    # Display final results
    print("\nFinal results:")
    print(json.dumps(final_data, indent=2))

    # Cleanup
    output_path.unlink(missing_ok=True)
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
