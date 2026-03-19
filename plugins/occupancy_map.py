"""
OccupancyMap — SLAM-lite incremental 3D occupancy grid plugin.

Subscribes to ``sensor.distance_3d`` and ``sensor.lidar`` events.
Builds and maintains a discretized occupancy grid that agents can
query for obstacle-free planning.

Cell states:
  -1 = unknown
   0 = free
   1 = occupied
"""

from __future__ import annotations

import math
from collections import defaultdict

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin

_UNKNOWN = -1
_FREE = 0
_OCCUPIED = 1


class OccupancyMap(BasePlugin):
    """
    Maintains a sparse 3D occupancy grid updated by sensor events.

    Parameters
    ----------
    event_bus : EventBus
    resolution : float
        Grid cell size in world units. Smaller = finer but more memory.
    max_range : float
        Ignore readings beyond this distance (they may be noise).
    decay_enabled : bool
        If True, occupied cells not re-observed after *decay_steps*
        updates are downgraded to unknown (handles dynamic obstacles).
    decay_steps : int
        Number of updates before stale occupied cells decay.
    """

    def __init__(
        self,
        event_bus: EventBus,
        resolution: float = 1.0,
        max_range: float = 15.0,
        decay_enabled: bool = False,
        decay_steps: int = 50,
    ) -> None:
        super().__init__(event_bus, plugin_id="occupancy_map")
        self._resolution = max(0.1, resolution)
        self._max_range = max_range
        self._decay_enabled = decay_enabled
        self._decay_steps = max(1, decay_steps)

        self._grid: dict[tuple[int, int, int], int] = defaultdict(lambda: _UNKNOWN)
        self._last_seen: dict[tuple[int, int, int], int] = {}
        self._update_count = 0
        self._total_updates = 0

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def grid(self) -> dict[tuple[int, int, int], int]:
        return dict(self._grid)

    def _world_to_grid(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            round(x / self._resolution),
            round(y / self._resolution),
            round(z / self._resolution),
        )

    def _grid_to_world(self, gx: int, gy: int, gz: int) -> tuple[float, float, float]:
        return (
            gx * self._resolution,
            gy * self._resolution,
            gz * self._resolution,
        )

    def get_cell(self, x: float, y: float, z: float) -> int:
        """Query cell state at world coordinates."""
        return self._grid[self._world_to_grid(x, y, z)]

    def is_free(self, x: float, y: float, z: float) -> bool:
        return self.get_cell(x, y, z) == _FREE

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        return self.get_cell(x, y, z) == _OCCUPIED

    def mark_free(self, gx: int, gy: int, gz: int) -> None:
        key = (gx, gy, gz)
        self._grid[key] = _FREE
        self._last_seen[key] = self._update_count

    def mark_occupied(self, gx: int, gy: int, gz: int) -> None:
        key = (gx, gy, gz)
        self._grid[key] = _OCCUPIED
        self._last_seen[key] = self._update_count

    def get_occupied_cells(self) -> list[tuple[int, int, int]]:
        return [k for k, v in self._grid.items() if v == _OCCUPIED]

    def get_free_cells(self) -> list[tuple[int, int, int]]:
        return [k for k, v in self._grid.items() if v == _FREE]

    def cell_count(self) -> dict[str, int]:
        unknown = sum(1 for v in self._grid.values() if v == _UNKNOWN)
        free = sum(1 for v in self._grid.values() if v == _FREE)
        occupied = sum(1 for v in self._grid.values() if v == _OCCUPIED)
        return {"unknown": unknown, "free": free, "occupied": occupied, "total": len(self._grid)}

    def start(self) -> None:
        super().start()
        self._bus.subscribe("sensor.distance_3d", self._on_distance)
        self._bus.subscribe("sensor.lidar", self._on_lidar)

    def stop(self) -> None:
        super().stop()
        self._bus.unsubscribe("sensor.distance_3d", self._on_distance)
        self._bus.unsubscribe("sensor.lidar", self._on_lidar)

    def _on_distance(self, event: dict) -> None:
        data = event.get("data", {})
        robot_pos = data.get("robot_position")
        distances = data.get("distances", {})
        if not robot_pos:
            return

        _DIRS = {
            "forward": (1, 0, 0), "backward": (-1, 0, 0),
            "left": (0, 1, 0), "right": (0, -1, 0),
            "up": (0, 0, 1), "down": (0, 0, -1),
        }

        for direction, info in distances.items():
            dist = info.get("distance", self._max_range)
            if dist > self._max_range:
                continue
            dx, dy, dz = _DIRS.get(direction, (0, 0, 0))
            if dx == 0 and dy == 0 and dz == 0:
                continue

            self._trace_ray(robot_pos, dx, dy, dz, dist)

        self._update_count += 1
        self._total_updates += 1
        self._apply_decay()

    def _on_lidar(self, event: dict) -> None:
        data = event.get("data", {})
        robot_pos = data.get("robot_position")
        if not robot_pos:
            return

        scans = data.get("scan") or []
        if not scans and "layers" in data:
            for layer in data["layers"]:
                scans.extend(layer)

        for ray in scans:
            angle = ray.get("angle_rad", 0)
            dist = ray.get("distance", self._max_range)
            hit = ray.get("hit", False)
            if dist > self._max_range:
                continue

            dx = math.cos(angle)
            dy = math.sin(angle)
            dz = 0.0

            self._trace_ray(robot_pos, dx, dy, dz, dist, hit_detected=hit)

        self._update_count += 1
        self._total_updates += 1
        self._apply_decay()

    def _trace_ray(
        self,
        origin: list[float],
        dx: float, dy: float, dz: float,
        distance: float,
        hit_detected: bool = True,
    ) -> None:
        """Mark cells along the ray as free, and the endpoint as occupied if hit."""
        steps = max(1, int(distance / self._resolution))
        step_size = distance / steps

        for i in range(steps):
            d = step_size * i
            wx = origin[0] + dx * d
            wy = origin[1] + dy * d
            wz = origin[2] + dz * d
            gx, gy, gz = self._world_to_grid(wx, wy, wz)
            self.mark_free(gx, gy, gz)

        if hit_detected and distance < self._max_range:
            ex = origin[0] + dx * distance
            ey = origin[1] + dy * distance
            ez = origin[2] + dz * distance
            gx, gy, gz = self._world_to_grid(ex, ey, ez)
            self.mark_occupied(gx, gy, gz)

    def _apply_decay(self) -> None:
        if not self._decay_enabled:
            return
        stale = []
        for key, last in self._last_seen.items():
            if self._grid[key] == _OCCUPIED and (self._update_count - last) > self._decay_steps:
                stale.append(key)
        for key in stale:
            self._grid[key] = _UNKNOWN

    def on_update(self) -> None:
        counts = self.cell_count()
        self._bus.publish(_make_event("map.occupancy", self._plugin_id, {
            "update_count": self._update_count,
            "cells": counts,
            "resolution": self._resolution,
        }))

    def clear(self) -> None:
        self._grid.clear()
        self._last_seen.clear()
        self._update_count = 0

    def get_summary(self) -> dict:
        counts = self.cell_count()
        return {
            "resolution": self._resolution,
            "update_count": self._update_count,
            "total_updates": self._total_updates,
            **counts,
        }
