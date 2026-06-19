"""NX6 — per-agent visibility view over a shared blackboard.

A non-invasive proxy: delegates EVERY blackboard method/attribute to the real
blackboard, except `get_artifacts_for_task`, which is filtered so the owning
agent only "sees" artifacts whose author is in its allowed set (its topology
neighbourhood). This realises a communication TOPOLOGY as a pure read-filter
— no agent code changes — exactly the clean knob NX6 needs.

Usage: give an agent `BlackboardView(real_bb, agent_id, visible_authors)` as
its blackboard. Posting, signals, claims, logging all pass straight through;
only what the agent can READ is restricted.
"""
from typing import Optional, Set
from .models import ArtifactType


class BlackboardView:
    def __init__(self, real_bb, agent_id: str, visible_authors: Optional[Set[str]] = None):
        object.__setattr__(self, "_bb", real_bb)
        object.__setattr__(self, "_aid", agent_id)
        # None  -> full visibility (sees everything, incl. own posts)
        # set   -> sees only artifacts authored by ids in this set (+ always self)
        object.__setattr__(self, "_vis", visible_authors)

    # the only filtered method
    def get_artifacts_for_task(self, task_id: str,
                               artifact_type: Optional[ArtifactType] = None):
        arts = self._bb.get_artifacts_for_task(task_id, artifact_type)
        vis = self._vis
        if vis is None:
            return arts
        allowed = set(vis) | {self._aid}
        return [a for a in arts if a.author in allowed or a.author == "system"]

    # everything else delegates transparently
    def __getattr__(self, name):
        return getattr(self._bb, name)
