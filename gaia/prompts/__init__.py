"""Prompt templates for GAIA agents"""

from .coder import CoderPrompts
from .critic import CriticPrompts
from .verifier import VerifierPrompts
from .planner import PlannerPrompts

__all__ = ["CoderPrompts", "CriticPrompts", "VerifierPrompts", "PlannerPrompts"]
