"""
PhysicsCamera — 3D camera sensor plugin using PyBullet's synthetic camera.
Captures depth data from the robot's perspective.
Supports configurable noise via NoiseMixin and publishes full depth matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pybullet as p

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin
from plugins.noise_mixin import NoiseMixin


class PhysicsCamera(NoiseMixin, BasePlugin):
    """Publishes ``sensor.camera_3d`` events with depth image data."""

    def __init__(
        self,
        event_bus: EventBus,
        physics_world: PhysicsWorld,
        width: int = 64,
        height: int = 64,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 20.0,
        noise_level: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(event_bus, plugin_id="physics_camera")
        self._world = physics_world
        if width < 1 or height < 1:
            raise ValueError("Camera width and height must be >= 1")
        self._img_w = width
        self._img_h = height
        self._fov = fov
        self._near = near
        self._far = far
        self._init_noise(noise_level, seed)

    def on_update(self) -> None:
        pos = self._world.get_robot_position()

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[pos[0], pos[1], pos[2] + 0.3],
            cameraTargetPosition=[pos[0] + 1, pos[1], pos[2] + 0.3],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._world._client,
        )
        aspect = self._img_w / self._img_h
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._fov, aspect=aspect,
            nearVal=self._near, farVal=self._far,
            physicsClientId=self._world._client,
        )

        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=self._img_w,
            height=self._img_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            physicsClientId=self._world._client,
        )

        depth_array = np.array(depth_buffer, dtype=np.float32).reshape(self._img_h, self._img_w)

        if self._noise > 0:
            noise_matrix = np.array(
                [[self._rng.gauss(0, self._noise) for _ in range(self._img_w)]
                 for _ in range(self._img_h)],
                dtype=np.float32,
            )
            raw_depth = depth_array.tolist()
            depth_array = np.clip(depth_array + noise_matrix, 0.0, 1.0)
        else:
            raw_depth = None

        event_data: dict = {
            "robot_position": list(pos),
            "depth": depth_array.tolist(),
            "image_size": [self._img_w, self._img_h],
            "noisy": self._noise > 0,
        }
        if raw_depth is not None:
            event_data["raw_depth"] = raw_depth

        self._bus.publish(_make_event("sensor.camera_3d", self._plugin_id, event_data))
