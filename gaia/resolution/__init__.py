"""Conflict resolution mechanisms for GAIA"""

from .conflict import ConflictDetector
from .branch_merge import BranchManager

__all__ = ["ConflictDetector", "BranchManager"]
