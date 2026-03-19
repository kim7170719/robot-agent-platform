"""
MockPhysicsWorld — in-memory stand-in for PhysicsWorld (no PyBullet).

Use for fast API/integration tests and environments where PyBullet is unavailable.
Implements the same public surface relied on by ``api.server`` and typical plugins.
"""

from __future__ import annotations

import math
from typing import Any

from simulator.physics_world import DIRECTIONS_3D

_ROBOT_RADIUS = 0.3
_OBSTACLE_HALF_EXTENT = 0.5


class MockPhysicsWorld:
    """Lightweight multi-robot world with obstacles (no physics engine)."""

    def __init__(self) -> None:
        self._obstacle_ids: list[int] = []
        self._obs_positions: dict[int, tuple[float, float, float]] = {}
        self._next_obstacle_id = 1
        self._robots: dict[str, tuple[float, float, float]] = {
            "default": (0.0, 0.0, _ROBOT_RADIUS + 0.01),
        }
        self._gui = False
        self._client = -1
        self._dynamic_ids: list[int] = []
        self._dyn_positions: dict[int, tuple[float, float, float]] = {}
        self._next_dynamic_id = 1

    @property
    def client_id(self) -> int:
        return self._client

    @property
    def robot_count(self) -> int:
        return len(self._robots)

    def add_robot(
        self, robot_id: str, x: float = 0, y: float = 0, z: float | None = None,
    ) -> int:
        if robot_id in self._robots:
            raise ValueError(f"Robot '{robot_id}' already exists")
        if z is None:
            z = _ROBOT_RADIUS + 0.01
        self._robots[robot_id] = (float(x), float(y), float(z))
        return hash(robot_id) % 10_000_000

    def remove_robot(self, robot_id: str) -> None:
        if robot_id == "default":
            raise ValueError("Cannot remove the default robot")
        if robot_id not in self._robots:
            raise ValueError(f"Robot '{robot_id}' not found")
        del self._robots[robot_id]

    def get_robot_ids(self) -> list[str]:
        return list(self._robots.keys())

    def _resolve_robot(self, robot_id: str | None) -> str:
        if robot_id is None:
            return "default"
        if robot_id not in self._robots:
            raise KeyError(f"Unknown robot '{robot_id}'. Available: {list(self._robots)}")
        return robot_id

    def place_obstacle(self, x: float, y: float, z: float = 0.5) -> int:
        oid = self._next_obstacle_id
        self._next_obstacle_id += 1
        self._obstacle_ids.append(oid)
        self._obs_positions[oid] = (float(x), float(y), float(z))
        return oid

    def clear_obstacles(self) -> None:
        self._obstacle_ids.clear()
        self._obs_positions.clear()

    def spawn_dynamic_box(
        self,
        x: float,
        y: float,
        z: float,
        *,
        half_extent: float = 0.1,
        mass: float = 1.0,
    ) -> int:
        del half_extent, mass  # mock: no physics; pose stored for API parity
        oid = self._next_dynamic_id
        self._next_dynamic_id += 1
        self._dynamic_ids.append(oid)
        self._dyn_positions[oid] = (float(x), float(y), float(z))
        return oid

    def clear_dynamic_objects(self) -> None:
        self._dynamic_ids.clear()
        self._dyn_positions.clear()

    def get_dynamic_objects(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bid in self._dynamic_ids:
            pos = self._dyn_positions.get(bid, (0.0, 0.0, 0.0))
            out.append({
                "id": bid,
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "linear_velocity": [0.0, 0.0, 0.0],
            })
        return out

    def get_obstacles(self) -> list[dict[str, Any]]:
        he = float(_OBSTACLE_HALF_EXTENT)
        out: list[dict[str, Any]] = []
        for oid in self._obstacle_ids:
            pos = self._obs_positions.get(oid, (0.0, 0.0, 0.5))
            out.append({
                "position": [round(pos[0], 4), round(pos[1], 4), round(pos[2], 4)],
                "half_extent": he,
            })
        return out

    def get_robot_position(self, robot_id: str | None = None) -> tuple[float, float, float]:
        rid = self._resolve_robot(robot_id)
        p = self._robots[rid]
        return (round(p[0], 4), round(p[1], 4), round(p[2], 4))

    def move_robot(self, direction: str, robot_id: str | None = None) -> bool:
        delta = DIRECTIONS_3D.get(direction)
        if delta is None:
            raise ValueError(f"Unknown direction: {direction!r}")
        rid = self._resolve_robot(robot_id)
        x, y, z = self._robots[rid]
        step = 0.15
        nx = x + delta[0] * step
        ny = y + delta[1] * step
        nz = z + delta[2] * step
        self._robots[rid] = (nx, ny, nz)
        return True

    def is_obstacle(self, x: float, y: float, z: float = 0) -> bool:
        for oid in self._obstacle_ids:
            ox, oy, oz = self._obs_positions.get(oid, (0.0, 0.0, 0.0))
            he = _OBSTACLE_HALF_EXTENT
            if abs(x - ox) <= he and abs(y - oy) <= he and abs(z - oz) <= he:
                return True
        return False

    def get_surroundings(self, x: float, y: float, z: float = 0, radius: int = 1) -> dict:
        pos = (float(x), float(y), float(z))
        result: dict[str, dict] = {}
        for name, (dx, dy, dz) in DIRECTIONS_3D.items():
            blocked = False
            distance = 20.0
            for oid in self._obstacle_ids:
                ox, oy, oz = self._obs_positions.get(oid, (0.0, 0.0, 0.0))
                hit = (ox, oy, oz)
                d = math.dist(pos, hit)
                if d < distance:
                    distance = d
                    blocked = True
            result[name] = {"blocked": blocked, "distance": round(distance, 3)}
        return result

    def step_physics(self) -> None:
        pass

    def render(self) -> str:
        lines = ["MockPhysicsWorld (no PyBullet)", f"  Robots: {len(self._robots)}"]
        for name in self._robots:
            pos = self.get_robot_position(name)
            lines.append(f"    [{name}] ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        lines.append(f"  Obstacles: {len(self._obstacle_ids)}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.clear_dynamic_objects()
        self.clear_obstacles()
        for i, name in enumerate(self._robots):
            self._robots[name] = (float(i * 2.0), 0.0, _ROBOT_RADIUS + 0.01)

    def set_robot_position(
        self, x: float, y: float, z: float | None = None, robot_id: str | None = None,
    ) -> None:
        if z is None:
            z = _ROBOT_RADIUS + 0.01
        rid = self._resolve_robot(robot_id)
        self._robots[rid] = (float(x), float(y), float(z))

    def disconnect(self) -> None:
        pass
