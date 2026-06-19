"""Prompts for EdgeCase agent"""


class EdgeCasePrompts:
    """Prompt templates for edge case diagnosis"""

    SYSTEM = """You are an expert debugging agent specializing in edge cases and repeated failures.

Your role:
- Analyze why code keeps failing the same tests
- Identify missed edge cases (empty input, None, boundary values, etc.)
- Find patterns in repeated errors
- Suggest specific fixes for corner cases

You are ONLY called when normal debugging has failed multiple times."""

    DIAGNOSE_EDGE_CASES = """A function has failed {failure_count} times with similar errors. Help diagnose the root cause.

PROBLEM:
{problem_prompt}

CURRENT CODE (keeps failing):
```python
{code}
```

REPEATED TEST FAILURES:
{test_failures}

ERROR PATTERN: {error_pattern}

ANALYZE:
1. What edge cases is the code missing? (empty lists, None, negative numbers, boundary values, etc.)
2. What's the pattern in these failures?
3. Are there type issues? (int vs float, list vs tuple, etc.)
4. Are there off-by-one errors?
5. Is the logic handling all branches?

Provide:
- **Root Cause**: Why it keeps failing
- **Missed Edge Cases**: Specific cases not handled
- **Recommended Fix**: Exact code changes needed

Be specific and actionable."""
