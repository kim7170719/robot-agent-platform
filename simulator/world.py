"""
GridWorld — 2D grid-based virtual environment for the robot.
Pure state container with no I/O and no external dependencies.
Implements WorldInterface for interchangeability with PhysicsWorld.
"""

from __future__ import annotations

import random

from simulator.world_interface import WorldInterface


DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class GridWorld(WorldInterface):
    """
    A width x height grid.  Cells are either free (0) or obstacle (1).
    The robot occupies a single cell.
    """

    def __init__(self, width: int = 10, height: int = 10) -> None:
        if width < 1 or height < 1:
            raise ValueError("World dimensions must be >= 1")
        self.width = width
        self.height = height
        self._grid: list[list[int]] = [
            [0] * width for _ in range(height)
        ]
        self._robot_pos: tuple[int, int] = (0, 0)

    def place_obstacle(self, x: int, y: int) -> None:
        if not self._in_bounds(x, y):
            raise IndexError(f"({x}, {y}) is out of bounds")
        if (x, y) == self._robot_pos:
            raise ValueError("Cannot place obstacle on robot position")
        self._grid[y][x] = 1

    def remove_obstacle(self, x: int, y: int) -> None:
        if not self._in_bounds(x, y):
            raise IndexError(f"({x}, {y}) is out of bounds")
        self._grid[y][x] = 0

    def is_obstacle(self, x: float, y: float, z: float = 0) -> bool:
        ix, iy = int(x), int(y)
        if not self._in_bounds(ix, iy):
            return True
        return self._grid[iy][ix] == 1

    def get_robot_position(self) -> tuple[int, int]:
        return self._robot_pos

    def set_robot_position(self, x: int, y: int) -> None:
        if not self._in_bounds(x, y):
            raise IndexError(f"({x}, {y}) is out of bounds")
        if self.is_obstacle(x, y):
            raise ValueError(f"({x}, {y}) is an obstacle")
        self._robot_pos = (x, y)

    def move_robot(self, direction: str) -> bool:
        delta = DIRECTIONS.get(direction)
        if delta is None:
            raise ValueError(f"Unknown direction: {direction!r}")
        nx, ny = self._robot_pos[0] + delta[0], self._robot_pos[1] + delta[1]
        if not self._in_bounds(nx, ny) or self.is_obstacle(nx, ny):
            return False
        self._robot_pos = (nx, ny)
        return True

    def get_obstacle_map(self) -> list[list[int]]:
        return [row[:] for row in self._grid]

    def get_surroundings(self, x: float, y: float, z: float = 0, radius: int = 1) -> dict:
        ix, iy = int(x), int(y)
        if radius < 1:
            return {name: {"blocked": False, "distance": 1} for name in DIRECTIONS}
        result = {}
        for name, (dx, dy) in DIRECTIONS.items():
            dist = radius + 1
            for step in range(1, radius + 1):
                if self.is_obstacle(ix + dx * step, iy + dy * step):
                    dist = step
                    break
            result[name] = {"blocked": dist <= radius, "distance": dist}
        return result

    def step_physics(self) -> None:
        pass  # no physics in grid world

    def render(self) -> str:
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self._robot_pos:
                    row += "R "
                elif self._grid[y][x] == 1:
                    row += "# "
                else:
                    row += ". "
            rows.append(row.rstrip())
        return "\n".join(rows)

    def reset(self) -> None:
        self._grid = [[0] * self.width for _ in range(self.height)]
        self._robot_pos = (0, 0)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


# Backward-compatible alias
World = GridWorld


def random_world(
    width: int = 10,
    height: int = 10,
    obstacle_ratio: float = 0.2,
    seed: int | None = None,
) -> GridWorld:
    if not 0.0 <= obstacle_ratio < 1.0:
        raise ValueError("obstacle_ratio must be in [0.0, 1.0)")
    rng = random.Random(seed)
    world = GridWorld(width, height)
    for y in range(height):
        for x in range(width):
            if (x, y) == (0, 0):
                continue
            if rng.random() < obstacle_ratio:
                world.place_obstacle(x, y)
    return world
