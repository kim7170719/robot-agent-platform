"""
GoalPublisher — plugin that publishes goal.target events to the EventBus.
Allows external systems to set navigation targets for agents.
No direct coupling: agents subscribe to goal events independently.
"""

from __future__ import annotations

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin


class GoalPublisher(BasePlugin):
    """
    Manages a target position and publishes ``goal.target`` events each update.
    Set the target via ``set_target()``; clear via ``clear_target()``.
    """

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus, plugin_id="goal_publisher")
        self._target: tuple[float, float, float] | None = None
        self._reached = False
        self._bus.subscribe("goal.reached", self._on_goal_reached)

    def _on_goal_reached(self, event: dict) -> None:
        self._reached = True

    @property
    def target(self) -> tuple[float, float, float] | None:
        return self._target

    @property
    def reached(self) -> bool:
        return self._reached

    def set_target(self, x: float, y: float, z: float = 0.31) -> None:
        self._target = (x, y, z)
        self._reached = False

    def clear_target(self) -> None:
        self._target = None
        self._reached = False

    def mark_reached(self) -> None:
        self._reached = True

    def on_update(self) -> None:
        if self._target is None:
            return
        self._bus.publish(_make_event("goal.target", self._plugin_id, {
            "target": list(self._target),
            "reached": self._reached,
        }))
