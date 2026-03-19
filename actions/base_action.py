"""
ActionLayer ABC — the bridge between agent decisions and world execution.

Agents call ``execute(command)`` instead of directly interacting with
the world.  This decoupling allows swapping SimulatedAction for a
real hardware adapter without changing agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ActionLayer(ABC):
    """
    Abstract action layer.

    ``command`` is a dict with at least ``{"direction": str}``.
    Returns a result dict with at least ``{"success": bool}``.
    """

    @abstractmethod
    def execute(self, command: dict) -> dict:
        """Translate a high-level command into world interaction and return result."""
