"""
PhysicsLidar — 3D LiDAR sensor plugin using PyBullet batch ray casting.
Simulates a planar laser sweep (configurable number of rays in 360 degrees).
Supports multi-layer scanning and configurable noise via NoiseMixin.
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


class PhysicsLidar(NoiseMixin, BasePlugin):
    """
    Publishes ``sensor.lidar`` events with a ring of distance measurements.
    Each scan contains ``num_rays`` evenly-spaced rays in the horizontal plane.
    """

    def __init__(
        self,
        event_bus: EventBus,
        physics_world: PhysicsWorld,
        num_rays: int = 36,
        max_distance: float = 15.0,
        height_offset: float = 0.3,
        num_layers: int = 1,
        layer_pitch_deg: float = 10.0,
        noise_level: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(event_bus, plugin_id="physics_lidar")
        self._world = physics_world
        self._num_rays = max(4, num_rays)
        self._max_dist = max_distance
        self._height_offset = height_offset
        self._num_layers = max(1, num_layers)
        self._layer_pitch = math.radians(layer_pitch_deg)
        self._init_noise(noise_level, seed)

    def on_update(self) -> None:
        pos = self._world.get_robot_position()
        origin_z = pos[2] + self._height_offset

        layers: list[list[dict]] = []

        for layer_i in range(self._num_layers):
            pitch = (layer_i - (self._num_layers - 1) / 2) * self._layer_pitch
            cos_p = math.cos(pitch)
            sin_p = math.sin(pitch)

            ray_from_list = []
            ray_to_list = []
            angles = []

            for i in range(self._num_rays):
                angle = 2 * math.pi * i / self._num_rays
                angles.append(angle)
                dx = math.cos(angle) * cos_p
                dy = math.sin(angle) * cos_p
                dz = sin_p

                ray_from_list.append([pos[0], pos[1], origin_z])
                ray_to_list.append([
                    pos[0] + dx * self._max_dist,
                    pos[1] + dy * self._max_dist,
                    origin_z + dz * self._max_dist,
                ])

            results = p.rayTestBatch(
                ray_from_list, ray_to_list,
                physicsClientId=self._world._client,
            )

            scan: list[dict] = []
            for i, hit in enumerate(results):
                hit_id = hit[0]
                hit_fraction = hit[2]

                if hit_id >= 0 and hit_id != self._world._robot_id:
                    raw_dist = round(hit_fraction * self._max_dist, 3)
                    hit_detected = True
                else:
                    raw_dist = round(self._max_dist, 3)
                    hit_detected = False

                scan.append({
                    "angle_rad": round(angles[i], 4),
                    "distance": self._apply_noise(raw_dist) if self._noise > 0 else raw_dist,
                    "raw_distance": raw_dist,
                    "hit": hit_detected,
                })
            layers.append(scan)

        event_data: dict = {
            "robot_position": list(pos),
            "num_rays": self._num_rays,
            "num_layers": self._num_layers,
            "max_distance": self._max_dist,
            "noisy": self._noise > 0,
        }
        if self._num_layers == 1:
            event_data["scan"] = layers[0]
        else:
            event_data["layers"] = layers

        self._bus.publish(_make_event("sensor.lidar", self._plugin_id, event_data))
