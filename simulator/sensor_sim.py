"""
SensorSim — generates mock sensor readings from World state.
Produces protocol-compliant event dicts for camera and distance sensors.
Supports configurable noise to simulate real-world sensor imprecision.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.world_interface import WorldInterface


class SensorSim:
    """
    Given a WorldInterface instance, synthesises sensor data that
    mock plugins can consume and publish to the Event Bus.

    When ``noise_level > 0``, distance readings are perturbed by
    gaussian noise (stddev = noise_level) and camera cells may
    randomly flip with probability = noise_level.
    """

    def __init__(self, world: WorldInterface, noise_level: float = 0.0, seed: int | None = None) -> None:
        self._world = world
        self._noise = max(0.0, noise_level)
        self._rng = random.Random(seed)

    @property
    def noise_level(self) -> float:
        return self._noise

    @noise_level.setter
    def noise_level(self, value: float) -> None:
        self._noise = max(0.0, value)

    def read_camera(self) -> dict:
        """
        Simulate a camera frame: returns a small grid patch
        around the robot (3x3 field of view).
        With noise, individual cells may randomly flip.
        """
        pos = self._world.get_robot_position()
        rx, ry = pos[0], pos[1]
        fov = {}
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                val = int(self._world.is_obstacle(rx + dx, ry + dy))
                if self._noise > 0 and self._rng.random() < self._noise:
                    val = 1 - val
                fov[f"{dx},{dy}"] = val

        return {
            "timestamp": int(time.time() * 1000),
            "type": "sensor.camera",
            "source": "sensor_sim",
            "data": {
                "robot_position": list(pos),
                "field_of_view": fov,
                "noisy": self._noise > 0,
            },
        }

    def read_distance(self) -> dict:
        """
        Simulate a distance sensor: returns distance to nearest
        obstacle in each cardinal direction (up/down/left/right).
        With noise, distance values are perturbed by gaussian noise.
        """
        pos = self._world.get_robot_position()
        rx, ry = pos[0], pos[1]
        surroundings = self._world.get_surroundings(rx, ry, radius=5)

        for info in surroundings.values():
            raw = info["distance"]
            info["raw_distance"] = raw
            if self._noise > 0:
                info["distance"] = round(max(1, raw + self._rng.gauss(0, self._noise)), 2)

        return {
            "timestamp": int(time.time() * 1000),
            "type": "sensor.distance",
            "source": "sensor_sim",
            "data": {
                "robot_position": list(pos),
                "distances": surroundings,
                "noisy": self._noise > 0,
            },
        }
