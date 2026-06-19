"""Parser for critic review output"""

import re
from typing import Tuple, Optional
from pydantic import BaseModel


class ReviewResult(BaseModel):
    """Parsed critic review"""

    is_agree: bool
    criticism: str = ""
    confidence: float = 0.5
    suggestions: str = ""


class ReviewParser:
    """Parse critic agent review output

    Pattern from AgentVerse's HumanevalCriticParser but more flexible.
    """

    @staticmethod
    def parse(text: str) -> ReviewResult:
        """Parse critic review

        Expected format:
        - Ends with [Agree] or [Disagree]
        - May contain criticism/feedback
        - May contain confidence level

        Args:
            text: Critic output text

        Returns:
            ReviewResult with parsed data
        """
        # Check for explicit agree/disagree markers
        is_agree = False

        if "[Agree]" in text or "[AGREE]" in text.upper():
            is_agree = True
        elif "[Disagree]" in text or "[DISAGREE]" in text.upper():
            is_agree = False
        else:
            # Heuristic: look for positive/negative language
            positive_words = ["correct", "looks good", "well done", "passes", "accurate"]
            negative_words = ["incorrect", "wrong", "fails", "error", "issue", "problem"]

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            is_agree = positive_count > negative_count

        # Extract criticism (remove agree/disagree markers)
        criticism = text.replace("[Agree]", "").replace("[Disagree]", "")
        criticism = criticism.replace("[AGREE]", "").replace("[DISAGREE]", "")
        criticism = criticism.strip()

        # Extract confidence if present (e.g., "Confidence: 0.8")
        confidence = 0.5
        confidence_match = re.search(r"confidence:?\s*(\d+\.?\d*)", criticism, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            # Normalize to 0-1 range if needed
            if confidence > 1.0:
                confidence = confidence / 100.0

        # Extract suggestions section if present
        suggestions = ""
        suggestions_match = re.search(
            r"(?:suggestions?|recommendations?|improvements?):\s*(.+?)(?=\n\n|\Z)",
            criticism,
            re.IGNORECASE | re.DOTALL
        )
        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()

        return ReviewResult(
            is_agree=is_agree,
            criticism=criticism,
            confidence=confidence,
            suggestions=suggestions,
        )

    @staticmethod
    def extract_issues(text: str) -> list[str]:
        """Extract list of specific issues mentioned

        Looks for:
        - Numbered lists of issues
        - Bullet points
        - "Issue: ..." patterns

        Args:
            text: Review text

        Returns:
            List of issue strings
        """
        issues = []

        # Look for "Issue:" or "Problem:" patterns
        issue_pattern = r"(?:Issue|Problem):\s*(.+?)(?=\n(?:Issue|Problem)|\n\n|\Z)"
        matches = re.findall(issue_pattern, text, re.IGNORECASE | re.DOTALL)
        issues.extend([m.strip() for m in matches])

        # Look for numbered lists within criticism sections
        if "criticism" in text.lower() or "issues" in text.lower():
            numbered = re.findall(r"\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)", text, re.DOTALL)
            issues.extend([n.strip() for n in numbered])

        return issues
