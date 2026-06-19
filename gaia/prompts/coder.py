"""Prompt templates for Coder agent"""


class CoderPrompts:
    """Prompts for code generation with team framing"""

    SYSTEM = """You are an expert Python programmer in a multi-agent team.
A Critic agent will review your code; a Fixer agent will apply targeted changes if needed.

Rules:
- Write the FULL function (signature + body + any needed imports at the top)
- Wrap your final code in ```python ... ``` markers
- Do not include test code — only the implementation
- If FIX_DIRECTIVES or FAILURE_HYPOTHESIS are provided, address each one precisely
- If MICRO-CHECKER WARNINGS are shown, pay close attention to those specific risk areas

Before finalising your code, mentally trace through these cases:
- Empty input / empty list / empty string → what does your function return?
- Negative numbers → does your function handle them correctly?
- The exact boundary value (e.g. n=0, n=1, shift == len(s)) → correct output?
If any trace reveals a bug, fix it before submitting.
"""

    INITIAL = """Complete the following Python function:

{problem_prompt}

Requirements:
- Provide ONLY the function implementation
- Ensure the code is correct and handles all edge cases
- Use clear variable names and logic
- The function must pass all test cases

Provide your solution in a ```python ... ``` code block."""

    RETRY_WITH_FEEDBACK = """The previous implementation failed tests. Here's the feedback:

{feedback}

Original problem:
{problem_prompt}

Previous attempt:
```python
{previous_code}
```

Please provide a corrected implementation that addresses the issues."""

    FIX_CONFLICT = """There is a conflict with your code. The critic identified the following issues:

{criticism}

Original problem:
{problem_prompt}

Your previous code:
```python
{previous_code}
```

Test results:
{test_output}

Please fix the code to address these issues and pass all tests."""
