"""Sandboxed Python code execution

Pattern from AgentVerse's CodeTestExecutor but simplified for HumanEval.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
import subprocess


class CodeRunner:
    """Execute Python code in subprocess with timeout

    Follows AgentVerse pattern but simplified since HumanEval provides test harness.
    """

    def __init__(self, timeout: int = 10, workdir: Optional[Path] = None):
        """
        Args:
            timeout: Execution timeout in seconds
            workdir: Working directory for code execution
        """
        self.timeout = timeout
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="gaia_sandbox_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    async def run_humaneval_test(
        self, code: str, test: str, entry_point: str
    ) -> Tuple[bool, str]:
        """Run HumanEval test for code

        Args:
            code: Generated function code
            test: Test harness from HumanEval (includes check() function)
            entry_point: Function name to test

        Returns:
            Tuple of (passed: bool, output: str)
        """
        # Construct test script:
        # 1. Generated code (function definition)
        # 2. Test harness from HumanEval (check function)
        # 3. Call to check(entry_point)
        test_script = f"""{code}

{test}

# Run the test
if __name__ == "__main__":
    try:
        check({entry_point})
        print("TESTS_PASSED")
    except AssertionError as e:
        print(f"TESTS_FAILED: {{e}}")
        exit(1)
    except Exception as e:
        print(f"ERROR: {{e}}")
        exit(2)
"""

        # Write to temp file
        test_file = self.workdir / "test_code.py"
        test_file.write_text(test_script)

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self._run_subprocess(test_file), timeout=self.timeout
            )
            passed = result.returncode == 0 and "TESTS_PASSED" in result.stdout
            output = result.stdout if result.stdout else result.stderr

            return passed, output

        except asyncio.TimeoutError:
            return False, f"Execution timed out after {self.timeout}s"
        except Exception as e:
            return False, f"Execution error: {str(e)}"

    async def _run_subprocess(self, script_path: Path) -> subprocess.CompletedProcess:
        """Run Python script in subprocess"""
        process = await asyncio.create_subprocess_exec(
            "python",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workdir),
        )

        stdout, stderr = await process.communicate()

        return subprocess.CompletedProcess(
            args=["python", str(script_path)],
            returncode=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    def cleanup(self):
        """Clean up working directory"""
        import shutil

        if self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)
