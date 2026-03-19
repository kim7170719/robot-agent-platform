"""
CollisionDetector — plugin that queries PyBullet contact points
each update and publishes ``physics.collision`` events to the EventBus.
Fully decoupled: agents subscribe to collision events independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pybullet as p

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin


class CollisionDetector(BasePlugin):
    """
    Publishes ``physics.collision`` events when any robot contacts
    an obstacle or the ground plane.

    Each event contains:
      - ``robot_id``: name of the robot involved
      - ``other_body``: "obstacle", "ground", or "robot:<id>"
      - ``contact_point``: [x, y, z] world-space position
      - ``normal_force``: scalar contact normal force
    """

    def __init__(
        self,
        event_bus: EventBus,
        physics_world: PhysicsWorld,
        include_ground: bool = False,
        force_threshold: float = 0.01,
    ) -> None:
        super().__init__(event_bus, plugin_id="collision_detector")
        self._world = physics_world
        self._include_ground = include_ground
        self._force_threshold = max(0.0, force_threshold)
        self._collision_count = 0

    @property
    def collision_count(self) -> int:
        return self._collision_count

    def on_update(self) -> None:
        p.performCollisionDetection(physicsClientId=self._world._client)

        body_to_name: dict[int, str] = {}
        for name, body_id in self._world._robots.items():
            body_to_name[body_id] = name

        robot_bodies = set(body_to_name.keys())
        obstacle_set = set(self._world._obstacle_ids)

        for robot_body, robot_name in body_to_name.items():
            contacts = p.getContactPoints(
                bodyA=robot_body, physicsClientId=self._world._client,
            )
            for cp in contacts:
                other_body = cp[2]

                if other_body == self._world._ground_id:
                    if not self._include_ground:
                        continue
                    other_label = "ground"
                elif other_body in obstacle_set:
                    other_label = "obstacle"
                elif other_body in robot_bodies:
                    other_label = f"robot:{body_to_name[other_body]}"
                else:
                    other_label = f"body:{other_body}"

                normal_force = abs(cp[9])
                if normal_force < self._force_threshold:
                    continue

                contact_pos = list(cp[6])

                self._collision_count += 1
                self._bus.publish(_make_event("physics.collision", self._plugin_id, {
                    "robot_id": robot_name,
                    "other_body": other_label,
                    "contact_point": [round(c, 4) for c in contact_pos],
                    "normal_force": round(normal_force, 4),
                }))
