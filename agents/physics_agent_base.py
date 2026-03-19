"""
PhysicsAgentBase — shared foundation for 3D physics agents.
Extracts visit memory, position quantization, obstacle-safe candidate
building, and the act-move-publish cycle that PhysicsAgent and GoalAgent
both duplicate.

Accepts an optional ``ActionLayer`` to decouple agent decisions from
world interaction.  When none is provided, a ``SimulatedAction`` is
created automatically so existing code works unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld
    from actions.base_action import ActionLayer

from agents.base_agent import BaseAgent
from core.event_bus import EventBus, _make_event
from simulator.physics_world import DIRECTIONS_3D

_SAFE_DISTANCE = 1.5


class PhysicsAgentBase(BaseAgent):
    """
    Base class for 3D physics agents with exploration memory.
    Subclasses must implement ``decide()``; everything else is provided.

    Supports multi-robot worlds via optional ``robot_id``.
    Supports pluggable ``ActionLayer`` for decoupled world interaction.
    """

    def __init__(
        self,
        event_bus: EventBus,
        world: PhysicsWorld,
        agent_id: str,
        robot_id: str | None = None,
        action_layer: ActionLayer | None = None,
    ) -> None:
        super().__init__(event_bus, agent_id=agent_id)
        self._world = world
        self._robot_id = robot_id
        self._latest_distance: dict | None = None
        self._chosen_action: str | None = None
        self._visit_count: dict[tuple[int, int, int], int] = defaultdict(int)

        if action_layer is None:
            from actions.simulated_action import SimulatedAction
            self._action_layer: ActionLayer = SimulatedAction(world, robot_id)
        else:
            self._action_layer = action_layer

        self._record_visit()

        self._bus.subscribe("sensor.distance_3d", self._on_distance)

    @property
    def visit_count(self) -> dict:
        return dict(self._visit_count)

    def cleanup(self) -> None:
        self._bus.unsubscribe("sensor.distance_3d", self._on_distance)

    def _quantize_pos(self) -> tuple[int, int, int]:
        """Discretize continuous position into grid cells for visit tracking."""
        pos = self._world.get_robot_position(self._robot_id)
        return (round(pos[0]), round(pos[1]), round(pos[2]))

    def _record_visit(self) -> None:
        self._visit_count[self._quantize_pos()] += 1

    def _on_distance(self, event: dict) -> None:
        src = event.get("data", {}).get("robot_id")
        if src is not None and self._robot_id is not None and src != self._robot_id:
            return
        self._latest_distance = event

    def perceive(self) -> None:
        pass

    def _safe_candidates(self) -> list[tuple[str, int, float]]:
        """
        Return a list of ``(direction, visit_count, neg_distance)`` for
        all directions where the obstacle distance exceeds _SAFE_DISTANCE.
        """
        distances = (self._latest_distance or {}).get("data", {}).get("distances", {})
        pos = self._world.get_robot_position(self._robot_id)
        result = []

        for direction, (dx, dy, dz) in DIRECTIONS_3D.items():
            dist = distances.get(direction, {}).get("distance", 0)
            if dist <= _SAFE_DISTANCE:
                continue
            target = (round(pos[0] + dx), round(pos[1] + dy), round(pos[2] + dz))
            result.append((direction, self._visit_count[target], -dist))
        return result

    def _build_move_event_data(self, pos_before: tuple) -> dict:
        """Standard payload for action.move_3d events."""
        data: dict = {
            "direction": self._chosen_action,
            "success": True,
            "position_before": list(pos_before),
            "new_position": list(self._world.get_robot_position(self._robot_id)),
        }
        if self._robot_id is not None:
            data["robot_id"] = self._robot_id
        return data

    @property
    def action_layer(self) -> ActionLayer:
        return self._action_layer

    def act(self) -> None:
        if self._chosen_action is None:
            return

        pos_before = self._world.get_robot_position(self._robot_id)
        result = self._action_layer.execute({"direction": self._chosen_action})
        success = result.get("success", False)

        if success:
            self._record_visit()

        event_data = self._build_move_event_data(pos_before)
        event_data["success"] = success
        self._bus.publish(_make_event("action.move_3d", self._agent_id, event_data))
        self._chosen_action = None
