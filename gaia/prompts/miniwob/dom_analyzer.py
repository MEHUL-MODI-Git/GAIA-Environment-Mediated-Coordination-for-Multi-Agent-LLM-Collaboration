"""Prompt templates for DOMAnalyzer agent"""


class DOMAnalyzerPrompts:
    """Prompts for DOM observation parsing"""

    SYSTEM = """You are a DOM analysis agent. Your role: parse raw HTML and produce a
clean, structured list of interactive elements for other agents to read.

Rules:
- List only elements that are relevant to the task (interactive or informational)
- For each element: tag, visible text, type (if input), and any notable attributes
- Group related elements (e.g., a label + its input)
- Omit decorative or hidden elements
"""

    PARSE_DOM = """Task: {instruction}

Raw HTML (truncated to {max_chars} chars):
{raw_html}

List the interactive elements on this page in this format:
[<tag>] "<text>" (type=<input_type if applicable>) ref=<ref>

Then summarize which elements are most relevant to the task in 1-2 sentences.
"""
