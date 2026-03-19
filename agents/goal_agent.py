"""
GoalAgent — 3D goal-directed agent with obstacle avoidance.
Subscribes to sensor.distance_3d and goal.target events.
Picks the direction that best reduces distance to the goal while
avoiding obstacles. Falls back to exploration when path is unclear.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld
    from actions.base_action import ActionLayer

from agents.physics_agent_base import PhysicsAgentBase
from core.event_bus import EventBus, _make_event
from simulator.physics_world import DIRECTIONS_3D

_GOAL_REACHED_RADIUS = 0.8


class GoalAgent(PhysicsAgentBase):
    """
    Goal-directed 3D agent. When a goal.target event is active,
    picks directions that minimize Euclidean distance to goal.
    Without a goal, defaults to exploration behavior.
    """

    def __init__(
        self,
        event_bus: EventBus,
        world: PhysicsWorld,
        robot_id: str | None = None,
        action_layer: ActionLayer | None = None,
    ) -> None:
        super().__init__(
            event_bus, world, agent_id="goal_agent",
            robot_id=robot_id, action_layer=action_layer,
        )
        self._goal: tuple[float, float, float] | None = None
        self._goal_reached = False
        self._bus.subscribe("goal.target", self._on_goal)

    @property
    def goal(self) -> tuple[float, float, float] | None:
        return self._goal

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    def cleanup(self) -> None:
        super().cleanup()
        self._bus.unsubscribe("goal.target", self._on_goal)

    def _on_goal(self, event: dict) -> None:
        target_data = event.get("data", {}).get("target")
        if target_data and len(target_data) >= 3:
            self._goal = (target_data[0], target_data[1], target_data[2])
            self._goal_reached = event.get("data", {}).get("reached", False)

    def _check_goal_reached(self) -> bool:
        if self._goal is None:
            return False
        pos = self._world.get_robot_position(self._robot_id)
        return math.dist(pos, self._goal) < _GOAL_REACHED_RADIUS

    def decide(self) -> None:
        if self._latest_distance is None:
            self._chosen_action = "forward"
            return

        if self._check_goal_reached():
            self._goal_reached = True
            self._chosen_action = None
            self._bus.publish(_make_event("goal.reached", self._agent_id, {
                "goal": list(self._goal),
                "position": list(self._world.get_robot_position()),
            }))
            return

        candidates = self._safe_candidates()
        if not candidates:
            self._chosen_action = None
            return

        if self._goal is not None:
            pos = self._world.get_robot_position(self._robot_id)
            scored = []
            for direction, visits, neg_dist in candidates:
                dx, dy, dz = DIRECTIONS_3D[direction]
                projected = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                goal_dist = math.dist(projected, self._goal)
                scored.append((direction, goal_dist, visits, neg_dist))
            scored.sort(key=lambda c: (c[1], c[2], c[3]))
            self._chosen_action = scored[0][0]
        else:
            candidates.sort(key=lambda c: (c[1], c[2]))
            self._chosen_action = candidates[0][0]

    def _build_move_event_data(self, pos_before: tuple) -> dict:
        data = super()._build_move_event_data(pos_before)
        data["goal"] = list(self._goal) if self._goal else None
        data["goal_distance"] = (
            round(math.dist(self._world.get_robot_position(self._robot_id), self._goal), 3)
            if self._goal else None
        )
        return data
