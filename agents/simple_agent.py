"""
SimpleAgent — concrete agent with obstacle-avoidance and exploration memory.
Subscribes to sensor events, decides a movement direction, and acts.
Uses a visited-cell counter to prefer unexplored cells and avoid oscillation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.world_interface import WorldInterface

from agents.base_agent import BaseAgent
from core.event_bus import EventBus, _make_event
from simulator.world import DIRECTIONS

PREFERRED_DIRECTIONS = ["right", "down", "left", "up"]


class SimpleAgent(BaseAgent):
    """
    Reactive agent with exploration memory:
      perceive -> buffer latest sensor data
      decide   -> pick direction preferring unvisited, passable cells
      act      -> move robot in the world & publish action event
    """

    def __init__(self, event_bus: EventBus, world: WorldInterface) -> None:
        super().__init__(event_bus, agent_id="simple_agent")
        self._world = world
        self._latest_distance: dict | None = None
        self._chosen_action: str | None = None
        self._visit_count: dict[tuple[int, int], int] = defaultdict(int)
        self._visit_count[world.get_robot_position()] = 1

        self._bus.subscribe("sensor.distance", self._on_distance)

    @property
    def visit_count(self) -> dict[tuple[int, int], int]:
        return dict(self._visit_count)

    def cleanup(self) -> None:
        """Unsubscribe from bus to prevent callback leaks."""
        self._bus.unsubscribe("sensor.distance", self._on_distance)

    def _on_distance(self, event: dict) -> None:
        self._latest_distance = event

    def perceive(self) -> None:
        pass  # data arrives via event callback

    def decide(self) -> None:
        if self._latest_distance is None:
            self._chosen_action = PREFERRED_DIRECTIONS[0]
            return

        distances = self._latest_distance.get("data", {}).get("distances", {})
        rx, ry = self._world.get_robot_position()

        candidates = []
        for direction in PREFERRED_DIRECTIONS:
            info = distances.get(direction, {})
            if info.get("distance", 0) > 1:
                dx, dy = DIRECTIONS[direction]
                target = (rx + dx, ry + dy)
                candidates.append((direction, self._visit_count[target]))

        if not candidates:
            self._chosen_action = None
            return

        candidates.sort(key=lambda c: c[1])
        self._chosen_action = candidates[0][0]

    def act(self) -> None:
        if self._chosen_action is None:
            return

        success = self._world.move_robot(self._chosen_action)

        if success:
            self._visit_count[self._world.get_robot_position()] += 1

        self._bus.publish(_make_event("action.move", self._agent_id, {
            "direction": self._chosen_action,
            "success": success,
            "new_position": list(self._world.get_robot_position()),
        }))
        self._chosen_action = None
