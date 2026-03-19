"""
PhysicsDistance — 3D distance sensor plugin using PyBullet ray casting.
Casts rays in 6 directions from the robot and reports hit distances.
Supports configurable gaussian noise via NoiseMixin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pybullet as p

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin
from plugins.noise_mixin import NoiseMixin
from simulator.physics_world import DIRECTIONS_3D, _RAY_MAX_DISTANCE


class PhysicsDistance(NoiseMixin, BasePlugin):
    """Publishes ``sensor.distance_3d`` events with ray-cast distance data."""

    def __init__(
        self,
        event_bus: EventBus,
        physics_world: PhysicsWorld,
        noise_level: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(event_bus, plugin_id="physics_distance")
        self._world = physics_world
        self._init_noise(noise_level, seed)

    def on_update(self) -> None:
        pos = self._world.get_robot_position()
        distances = {}

        for name, (dx, dy, dz) in DIRECTIONS_3D.items():
            ray_from = list(pos)
            ray_to = [
                pos[0] + dx * _RAY_MAX_DISTANCE,
                pos[1] + dy * _RAY_MAX_DISTANCE,
                pos[2] + dz * _RAY_MAX_DISTANCE,
            ]
            hits = p.rayTest(ray_from, ray_to, physicsClientId=self._world._client)

            distance = _RAY_MAX_DISTANCE
            blocked = False
            for hit in hits:
                hit_id = hit[0]
                hit_pos = hit[3]
                if hit_id >= 0 and hit_id != self._world._robot_id:
                    distance = math.dist(pos, hit_pos)
                    blocked = True
                    break

            raw = round(distance, 3)
            distances[name] = {
                "blocked": blocked,
                "distance": self._apply_noise(raw) if self._noise > 0 else raw,
                "raw_distance": raw,
            }

        self._bus.publish(_make_event("sensor.distance_3d", self._plugin_id, {
            "robot_position": list(pos),
            "distances": distances,
            "noisy": self._noise > 0,
        }))
