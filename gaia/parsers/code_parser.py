"""Parser for extracting Python code from LLM output"""

import re
from typing import Optional


class CodeParser:
    """Extract Python code from markdown code blocks or raw text

    Pattern from AgentVerse's HumanevalSolverParser but more robust.
    """

    @staticmethod
    def parse(text: str) -> Optional[str]:
        """Extract code from LLM response

        Args:
            text: LLM response text

        Returns:
            Extracted code or None if no code found
        """
        # Try to extract from markdown code blocks first
        # Pattern: ```python ... ``` or ```\n...\n```
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)

        if code_blocks:
            # Return the last code block (most likely to be the final solution)
            return code_blocks[-1].strip()

        # Try to find code without markdown
        # Look for function definitions
        func_match = re.search(r"(def\s+\w+.*?)(?:\n\n|\Z)", text, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        # If no code blocks or function definitions, return the whole text
        # (might be inline code)
        return text.strip()

    @staticmethod
    def extract_function(text: str, function_name: str) -> Optional[str]:
        """Extract a specific function by name

        Args:
            text: Code text
            function_name: Name of function to extract

        Returns:
            Function code or None
        """
        pattern = rf"(def\s+{function_name}\s*\(.*?\):.*?)(?=\ndef\s|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def clean_code(code: str) -> str:
        """Clean code by removing common artifacts

        Args:
            code: Raw code string

        Returns:
            Cleaned code
        """
        # Remove markdown artifacts
        code = code.replace("```python", "").replace("```", "")

        # Remove common LLM artifacts
        code = re.sub(r"^(Here'?s? .*?:|Solution:)", "", code, flags=re.IGNORECASE)

        # Remove leading/trailing whitespace
        code = code.strip()

        return code
