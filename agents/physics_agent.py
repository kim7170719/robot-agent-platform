"""
PhysicsAgent — 3D obstacle-avoidance agent for PhysicsWorld.
Subscribes to sensor.distance_3d events, decides movement in 6 directions,
and uses exploration memory to avoid revisiting the same region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld
    from actions.base_action import ActionLayer

from agents.physics_agent_base import PhysicsAgentBase
from core.event_bus import EventBus

PREFERRED_DIRECTIONS_3D = ["forward", "right", "left", "backward", "up", "down"]


class PhysicsAgent(PhysicsAgentBase):
    """
    3D reactive agent with exploration memory.
    Prefers directions that are far from obstacles and unvisited.
    """

    def __init__(
        self,
        event_bus: EventBus,
        world: PhysicsWorld,
        robot_id: str | None = None,
        action_layer: ActionLayer | None = None,
    ) -> None:
        super().__init__(
            event_bus, world, agent_id="physics_agent",
            robot_id=robot_id, action_layer=action_layer,
        )

    def decide(self) -> None:
        if self._latest_distance is None:
            self._chosen_action = PREFERRED_DIRECTIONS_3D[0]
            return

        candidates = self._safe_candidates()
        if not candidates:
            self._chosen_action = None
            return

        candidates.sort(key=lambda c: (c[1], c[2]))
        self._chosen_action = candidates[0][0]
