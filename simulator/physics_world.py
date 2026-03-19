"""
PhysicsWorld — 3D physics-based environment using PyBullet.
Implements WorldInterface for interchangeability with GridWorld.
Default headless (DIRECT) mode requires no GPU.
"""

from __future__ import annotations

import math
from typing import Any

import pybullet as p
import pybullet_data

from simulator.world_interface import WorldInterface


DIRECTIONS_3D = {
    "forward":  (1, 0, 0),
    "backward": (-1, 0, 0),
    "left":     (0, 1, 0),
    "right":    (0, -1, 0),
    "up":       (0, 0, 1),
    "down":     (0, 0, -1),
}

_ROBOT_RADIUS = 0.3
_MOVE_FORCE = 200.0
_OBSTACLE_HALF_EXTENT = 0.5
_RAY_MAX_DISTANCE = 20.0


_ROBOT_COLORS = [
    [0.2, 0.6, 1.0, 1.0],
    [0.2, 1.0, 0.4, 1.0],
    [1.0, 0.8, 0.1, 1.0],
    [1.0, 0.3, 0.7, 1.0],
    [0.6, 0.3, 1.0, 1.0],
    [0.1, 0.9, 0.9, 1.0],
]


class PhysicsWorld(WorldInterface):
    """
    3D world backed by PyBullet's Bullet physics engine.

    - Default headless (``gui=False``) for testing / CI.
    - Fixed timestep for deterministic behaviour.
    - Robot is a sphere; obstacles are static boxes.
    - Supports multiple robots via ``add_robot()`` / ``get_robot_position(robot_id=...)``.
    """

    def __init__(
        self,
        gui: bool = False,
        timestep: float = 1 / 240,
        gravity: float = -9.81,
    ) -> None:
        mode = p.GUI if gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, gravity, physicsClientId=self._client)
        p.setTimeStep(timestep, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            enableConeFriction=1, physicsClientId=self._client
        )

        self._timestep = timestep
        self._gui = gui
        self._obstacle_ids: list[int] = []
        self._dynamic_ids: list[int] = []
        self._robots: dict[str, int] = {}

        self._ground_id = p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._robot_id = self._spawn_robot(
            [0, 0, _ROBOT_RADIUS + 0.01], _ROBOT_COLORS[0]
        )
        self._robots["default"] = self._robot_id

    def _spawn_robot(
        self, position: list[float], color: list[float],
    ) -> int:
        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=_ROBOT_RADIUS, physicsClientId=self._client
        )
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=_ROBOT_RADIUS,
            rgbaColor=color, physicsClientId=self._client
        )
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self._client,
        )
        p.changeDynamics(
            body_id, -1,
            linearDamping=0.9, angularDamping=0.9,
            physicsClientId=self._client,
        )
        return body_id

    @property
    def client_id(self) -> int:
        """Expose the PyBullet client ID for URDFRobot and other extensions."""
        return self._client

    def add_robot(
        self, robot_id: str, x: float = 0, y: float = 0, z: float | None = None,
    ) -> int:
        """Spawn an additional named robot. Returns PyBullet body id."""
        if robot_id in self._robots:
            raise ValueError(f"Robot '{robot_id}' already exists")
        if z is None:
            z = _ROBOT_RADIUS + 0.01
        color_idx = len(self._robots) % len(_ROBOT_COLORS)
        body = self._spawn_robot([x, y, z], _ROBOT_COLORS[color_idx])
        self._robots[robot_id] = body
        return body

    def remove_robot(self, robot_id: str) -> None:
        """Remove a named robot from the world."""
        if robot_id == "default":
            raise ValueError("Cannot remove the default robot")
        body = self._robots.pop(robot_id, None)
        if body is None:
            raise ValueError(f"Robot '{robot_id}' not found")
        p.removeBody(body, physicsClientId=self._client)

    def get_robot_ids(self) -> list[str]:
        return list(self._robots.keys())

    def _resolve_robot(self, robot_id: str | None) -> int:
        """Return PyBullet body id for the given robot_id, default to primary."""
        if robot_id is None:
            return self._robot_id
        if robot_id not in self._robots:
            raise KeyError(f"Unknown robot '{robot_id}'. Available: {list(self._robots)}")
        return self._robots[robot_id]

    # ── obstacle management ──────────────────────────────

    def place_obstacle(self, x: float, y: float, z: float = 0.5) -> int:
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[_OBSTACLE_HALF_EXTENT] * 3,
            physicsClientId=self._client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[_OBSTACLE_HALF_EXTENT] * 3,
            rgbaColor=[0.8, 0.2, 0.2, 1.0],
            physicsClientId=self._client,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z],
            physicsClientId=self._client,
        )
        self._obstacle_ids.append(body_id)
        return body_id

    def clear_obstacles(self) -> None:
        for oid in self._obstacle_ids:
            p.removeBody(oid, physicsClientId=self._client)
        self._obstacle_ids.clear()

    def spawn_dynamic_box(
        self,
        x: float,
        y: float,
        z: float,
        *,
        half_extent: float = 0.1,
        mass: float = 1.0,
    ) -> int:
        """
        Spawn a **dynamic** box (mass > 0) for drop tests, stacking, or pick-place probes.

        Static obstacles from ``place_obstacle`` do not move; these bodies are affected by
        gravity and collisions. IDs are tracked separately from ``_obstacle_ids``.
        """
        he = float(half_extent)
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[he, he, he], physicsClientId=self._client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[he, he, he],
            rgbaColor=[0.92, 0.55, 0.12, 1.0],
            physicsClientId=self._client,
        )
        bid = p.createMultiBody(
            baseMass=float(mass),
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[float(x), float(y), float(z)],
            physicsClientId=self._client,
        )
        self._dynamic_ids.append(bid)
        return bid

    def clear_dynamic_objects(self) -> None:
        for bid in self._dynamic_ids:
            try:
                p.removeBody(bid, physicsClientId=self._client)
            except Exception:
                pass
        self._dynamic_ids.clear()

    def get_dynamic_objects(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bid in self._dynamic_ids:
            pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=self._client)
            lin, _ = p.getBaseVelocity(bid, physicsClientId=self._client)
            out.append({
                "id": bid,
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "linear_velocity": [round(lin[0], 4), round(lin[1], 4), round(lin[2], 4)],
            })
        return out

    def get_obstacles(self) -> list[dict[str, Any]]:
        """Return obstacle centres and half-extent for visualization / API."""
        out: list[dict[str, Any]] = []
        he = float(_OBSTACLE_HALF_EXTENT)
        for oid in self._obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self._client)
            out.append({
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "half_extent": he,
            })
        return out

    # ── WorldInterface implementation ────────────────────

    def get_robot_position(self, robot_id: str | None = None) -> tuple[float, float, float]:
        body = self._resolve_robot(robot_id)
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self._client)
        return (round(pos[0], 4), round(pos[1], 4), round(pos[2], 4))

    def move_robot(self, direction: str, robot_id: str | None = None) -> bool:
        delta = DIRECTIONS_3D.get(direction)
        if delta is None:
            raise ValueError(f"Unknown direction: {direction!r}")
        body = self._resolve_robot(robot_id)
        force = [d * _MOVE_FORCE for d in delta]
        pos_before = self.get_robot_position(robot_id)
        p.applyExternalForce(
            body, -1, force, [0, 0, 0],
            p.WORLD_FRAME, physicsClientId=self._client,
        )
        for _ in range(60):
            p.stepSimulation(physicsClientId=self._client)
        pos_after = self.get_robot_position(robot_id)
        moved = math.dist(pos_before, pos_after) > 0.01
        return moved

    def is_obstacle(self, x: float, y: float, z: float = 0) -> bool:
        test_from = [x, y, z + 5.0]
        test_to = [x, y, z - 5.0]
        results = p.rayTest(test_from, test_to, physicsClientId=self._client)
        for hit in results:
            hit_id = hit[0]
            if hit_id in self._obstacle_ids:
                return True
        return False

    def get_surroundings(self, x: float, y: float, z: float = 0, radius: int = 1) -> dict:
        result = {}
        pos = (float(x), float(y), float(z) if z != 0 else self.get_robot_position()[2])
        for name, (dx, dy, dz) in DIRECTIONS_3D.items():
            ray_from = list(pos)
            ray_to = [
                pos[0] + dx * _RAY_MAX_DISTANCE,
                pos[1] + dy * _RAY_MAX_DISTANCE,
                pos[2] + dz * _RAY_MAX_DISTANCE,
            ]
            hits = p.rayTest(ray_from, ray_to, physicsClientId=self._client)
            distance = _RAY_MAX_DISTANCE
            blocked = False
            for hit in hits:
                hit_id, _, hit_fraction, hit_pos, _ = hit
                if hit_id >= 0 and hit_id != self._robot_id:
                    distance = math.dist(pos, hit_pos)
                    blocked = True
                    break
            result[name] = {"blocked": blocked, "distance": round(distance, 3)}
        return result

    def step_physics(self) -> None:
        p.stepSimulation(physicsClientId=self._client)

    def render(self) -> str:
        lines = [
            f"PhysicsWorld (PyBullet {'GUI' if self._gui else 'DIRECT'})",
            f"  Robots: {len(self._robots)}",
        ]
        for name in self._robots:
            pos = self.get_robot_position(name)
            lines.append(f"    [{name}] ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        lines.append(f"  Obstacles: {len(self._obstacle_ids)}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.clear_dynamic_objects()
        self.clear_obstacles()
        for i, (name, body) in enumerate(self._robots.items()):
            p.resetBasePositionAndOrientation(
                body,
                [i * 2.0, 0, _ROBOT_RADIUS + 0.01],
                [0, 0, 0, 1],
                physicsClientId=self._client,
            )
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

    def set_robot_position(
        self, x: float, y: float, z: float | None = None, robot_id: str | None = None,
    ) -> None:
        if z is None:
            z = _ROBOT_RADIUS + 0.01
        body = self._resolve_robot(robot_id)
        p.resetBasePositionAndOrientation(
            body, [x, y, z], [0, 0, 0, 1], physicsClientId=self._client,
        )
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

    @property
    def robot_count(self) -> int:
        return len(self._robots)

    def disconnect(self) -> None:
        if p.isConnected(self._client):
            p.disconnect(self._client)

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass
