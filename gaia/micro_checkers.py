"""Micro-checkers: zero-cost risk detection before any LLM call.

Scans the problem spec for patterns that historically cause bugs.
Injects targeted warnings into Coder and Critic prompts.
"""

from dataclasses import dataclass, field


@dataclass
class CheckerResult:
    checker_name: str
    triggered: bool
    warnings: list = field(default_factory=list)


@dataclass
class MicroCheckReport:
    results: list = field(default_factory=list)

    @property
    def any_triggered(self) -> bool:
        return any(r.triggered for r in self.results)

    @property
    def triggered_names(self) -> list:
        return [r.checker_name for r in self.results if r.triggered]

    def to_prompt_block(self) -> str:
        """Format triggered warnings as a prompt injection block."""
        triggered = [r for r in self.results if r.triggered]
        if not triggered:
            return ""

        lines = ["## ⚠️ MICRO-CHECKER WARNINGS (pay close attention to these)\n"]
        for r in triggered:
            lines.append(f"### 🔍 {r.checker_name}")
            for w in r.warnings:
                lines.append(f"  - {w}")
            lines.append("")
        return "\n".join(lines)

    def risk_flags(self) -> dict:
        """Return boolean flags for each risk category."""
        names = set(self.triggered_names)
        return {
            "boundary_risk": "THRESHOLD/BOUNDARY" in names,
            "tie_break_risk": "TIE_BREAK" in names,
            "digit_negative_risk": "DIGIT/NEGATIVE" in names,
            "regex_string_risk": "REGEX/STRING" in names,
            "sequence_risk": "SEQUENCE/RECURRENCE" in names,
        }


def _check_threshold(prompt: str) -> CheckerResult:
    """Detect boundary/threshold comparisons that commonly have off-by-one bugs."""
    keywords = [">=", "<=", "threshold", "tier", "grade", "rank", "at least",
                "between", "minimum", "maximum", "no more than", "no less than",
                "at most", "strictly", "inclusive", "exclusive"]
    lower = prompt.lower()
    triggered = any(k in lower for k in keywords)
    warnings = []
    if triggered:
        warnings = [
            "Prompt contains boundary/threshold language.",
            "CRITICAL: For EVERY comparison, determine EXACTLY whether it is > or >= (< or <=).",
            "Build a mental table: what happens at the exact boundary value?",
            "Write boundary tests: assert func(boundary-1) != func(boundary) if needed.",
        ]
    return CheckerResult("THRESHOLD/BOUNDARY", triggered, warnings)


def _check_tie_break(prompt: str) -> CheckerResult:
    """Detect tie-breaking / equality conditions that are easy to overlook."""
    keywords = ["tie", "equal", "same", "prefer", "stable sort", "first occurrence",
                "last occurrence", "earlier", "later", "lexicographically"]
    lower = prompt.lower()
    triggered = any(k in lower for k in keywords)
    warnings = []
    if triggered:
        warnings = [
            "Prompt contains tie-breaking or equality language.",
            "CRITICAL: What happens when two values are exactly equal?",
            "Check if 'first' means index 0 or 1-indexed.",
            "Use stable sort if order among equals must be preserved.",
            "Write test cases where multiple items tie.",
        ]
    return CheckerResult("TIE_BREAK", triggered, warnings)


def _check_digit_negative(prompt: str) -> CheckerResult:
    """Detect digit/negative number handling that is commonly mishandled."""
    keywords = ["digit", "sum of digit", "even digit", "odd digit", "negative",
                "sign", "absolute", "positive", "non-negative", "non-positive"]
    lower = prompt.lower()
    triggered = any(k in lower for k in keywords)
    warnings = []
    if triggered:
        warnings = [
            "Prompt contains digit or negative number language.",
            "CRITICAL: 'digit' means a single character 0-9, not the full number.",
            "Even digits are: 0, 2, 4, 6, 8 (zero IS even).",
            "For negative numbers: the minus sign is NOT a digit.",
            "Use abs() when computing digit sums of negative numbers.",
            "Test with: 0, negative numbers, single-digit numbers.",
        ]
    return CheckerResult("DIGIT/NEGATIVE", triggered, warnings)


def _check_regex_string(prompt: str) -> CheckerResult:
    """Detect string/regex parsing that is commonly done incorrectly."""
    keywords = ["split", "sentence", "word", "character", "pattern", "regex",
                "bracket", "nested", "starts with", "ends with", "last char",
                "first char", "substring", "subsequence", "palindrome", "delimiter",
                "whitespace", "punctuation"]
    lower = prompt.lower()
    triggered = any(k in lower for k in keywords)
    warnings = []
    if triggered:
        warnings = [
            "Prompt contains string/pattern parsing language.",
            "CRITICAL: distinguish subsequence (non-contiguous) vs substring (contiguous).",
            "Sentence delimiters are typically: '.', '?', '!'",
            "Check if splitting on whitespace vs single space matters.",
            "For bracket matching: use a stack, handle nested brackets correctly.",
            "Test with: empty string, single char, multiple spaces, edge delimiters.",
        ]
    return CheckerResult("REGEX/STRING", triggered, warnings)


def _check_sequence_recurrence(prompt: str) -> CheckerResult:
    """Detect sequence/recurrence problems where formula is easy to get wrong."""
    keywords = ["fibonacci", "tribonacci", "sequence", "recurrence", "recursion",
                "n-th", "nth", "odd index", "even index", "previous", "f(n)",
                "a(n)", "term", "series"]
    lower = prompt.lower()
    triggered = any(k in lower for k in keywords)
    warnings = []
    if triggered:
        warnings = [
            "Prompt contains sequence/recurrence language.",
            "CRITICAL: Copy the EXACT recurrence formula from the spec — do not assume.",
            "Compute the first 5-8 values by hand to verify your formula.",
            "Check: does the spec use 0-indexed or 1-indexed terms?",
            "Off-by-one: does 'n-th term' mean index n or the nth element?",
            "Watch for sequences that need n+1 elements to compute the nth term.",
        ]
    return CheckerResult("SEQUENCE/RECURRENCE", triggered, warnings)


def run_micro_checkers(prompt: str) -> MicroCheckReport:
    """Run all micro-checkers on the problem prompt.

    Args:
        prompt: The HumanEval function spec/docstring

    Returns:
        MicroCheckReport with results from all checkers
    """
    results = [
        _check_threshold(prompt),
        _check_tie_break(prompt),
        _check_digit_negative(prompt),
        _check_regex_string(prompt),
        _check_sequence_recurrence(prompt),
    ]
    report = MicroCheckReport(results=results)
    return report
