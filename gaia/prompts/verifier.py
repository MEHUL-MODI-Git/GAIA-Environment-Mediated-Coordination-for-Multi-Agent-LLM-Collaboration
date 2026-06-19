"""Prompt templates for Verifier agent"""


class VerifierPrompts:
    """Prompts for verification (mainly used for logging/explanations)"""

    SYSTEM = """You are a verification agent. Your task is to run tests and ensure code correctness through objective checks."""

    VERIFICATION_REPORT = """Verification Report for {task_id}

Code:
```python
{code}
```

Test Execution Results:
{test_output}

Status: {"✓ PASSED" if passed else "✗ FAILED"}

{additional_notes}"""
