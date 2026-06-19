"""Verification gate for GAIA - ensures artifact correctness"""

from typing import Tuple, Optional
from ..blackboard.models import Evidence
from .code_runner import CodeRunner


class VerificationGate:
    """Verification gate (part of Feature A)

    Only accepts artifacts that pass objective checks.
    For HumanEval: runs unit tests to verify code correctness.
    """

    def __init__(self, code_runner: Optional[CodeRunner] = None):
        self.code_runner = code_runner or CodeRunner()

    async def verify_code(
        self, code: str, test: str, entry_point: str
    ) -> Tuple[bool, Evidence]:
        """Verify code by running tests

        Args:
            code: Generated code to verify
            test: Test harness
            entry_point: Function name

        Returns:
            Tuple of (passed: bool, Evidence object)
        """
        # Run tests
        passed, output = await self.code_runner.run_humaneval_test(code, test, entry_point)

        # Create evidence
        evidence = Evidence(
            type="test_result",
            content=output,
            passed=passed,
            metadata={
                "entry_point": entry_point,
                "test_length": len(test),
                "code_length": len(code),
            },
        )

        return passed, evidence

    def check_policy_compliance(self, policy_name: str, artifact: any) -> bool:
        """Check if artifact complies with a policy

        Args:
            policy_name: Name of policy to check
            artifact: Artifact to verify

        Returns:
            True if compliant
        """
        # Extensible for future policies beyond test execution
        if policy_name == "all_tests_pass":
            # This is handled by verify_code
            return True

        return False
