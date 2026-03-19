"""
RRT* — Rapidly-exploring Random Tree (Star) for continuous 3D path planning.

Unlike A* which operates on a discretized grid, RRT* plans in continuous
space and naturally handles high-DOF problems and complex obstacle geometries.

Features:
  - Configurable workspace bounds and step size
  - Obstacle collision checking via callback
  - Near-optimal paths via rewiring (RRT* extension)
  - Path smoothing post-processing
  - Works with any callable collision checker
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RRTNode:
    position: tuple[float, float, float]
    parent: RRTNode | None = None
    cost: float = 0.0
    children: list[RRTNode] = field(default_factory=list)

    def __hash__(self) -> int:
        return id(self)


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _steer(
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    step_size: float,
) -> tuple[float, float, float]:
    d = _dist(from_pos, to_pos)
    if d <= step_size:
        return to_pos
    ratio = step_size / d
    return (
        from_pos[0] + (to_pos[0] - from_pos[0]) * ratio,
        from_pos[1] + (to_pos[1] - from_pos[1]) * ratio,
        from_pos[2] + (to_pos[2] - from_pos[2]) * ratio,
    )


def _interpolate_path(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    resolution: float = 0.1,
) -> list[tuple[float, float, float]]:
    d = _dist(a, b)
    if d < resolution:
        return [a, b]
    steps = max(2, int(d / resolution))
    path = []
    for i in range(steps + 1):
        t = i / steps
        path.append((
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ))
    return path


class RRTStar:
    """
    RRT* planner in 3D continuous space.

    Parameters
    ----------
    bounds_min, bounds_max : tuple
        Workspace bounding box.
    step_size : float
        Maximum extension distance per iteration.
    goal_radius : float
        Distance threshold to consider goal reached.
    max_iterations : int
    collision_fn : callable
        ``fn(x, y, z) -> bool`` returns True if the point is in collision.
    rewire_radius : float
        Radius for RRT* rewiring. Set 0 to disable (plain RRT).
    seed : int | None
    """

    def __init__(
        self,
        bounds_min: tuple[float, float, float] = (-10, -10, 0),
        bounds_max: tuple[float, float, float] = (10, 10, 5),
        step_size: float = 0.5,
        goal_radius: float = 0.5,
        max_iterations: int = 2000,
        collision_fn: Callable[[float, float, float], bool] | None = None,
        rewire_radius: float = 1.5,
        seed: int | None = None,
    ) -> None:
        self._bounds_min = bounds_min
        self._bounds_max = bounds_max
        self._step_size = step_size
        self._goal_radius = goal_radius
        self._max_iter = max_iterations
        self._collision_fn = collision_fn or (lambda x, y, z: False)
        self._rewire_radius = rewire_radius
        self._rng = random.Random(seed)
        self._nodes: list[RRTNode] = []
        self._iterations_used = 0

    @property
    def nodes(self) -> list[RRTNode]:
        return list(self._nodes)

    @property
    def iterations_used(self) -> int:
        return self._iterations_used

    def _random_point(self) -> tuple[float, float, float]:
        return (
            self._rng.uniform(self._bounds_min[0], self._bounds_max[0]),
            self._rng.uniform(self._bounds_min[1], self._bounds_max[1]),
            self._rng.uniform(self._bounds_min[2], self._bounds_max[2]),
        )

    def _nearest(self, point: tuple[float, float, float]) -> RRTNode:
        best = self._nodes[0]
        best_d = _dist(best.position, point)
        for node in self._nodes[1:]:
            d = _dist(node.position, point)
            if d < best_d:
                best = node
                best_d = d
        return best

    def _near_nodes(self, point: tuple[float, float, float]) -> list[RRTNode]:
        if self._rewire_radius <= 0:
            return []
        return [n for n in self._nodes if _dist(n.position, point) <= self._rewire_radius]

    def _collision_free_path(
        self, a: tuple[float, float, float], b: tuple[float, float, float]
    ) -> bool:
        for pt in _interpolate_path(a, b, resolution=self._step_size * 0.3):
            if self._collision_fn(pt[0], pt[1], pt[2]):
                return False
        return True

    def plan(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        """
        Plan a path from start to goal. Returns list of waypoints or
        empty list if no path found within max_iterations.
        """
        self._nodes = [RRTNode(position=start, cost=0.0)]
        self._iterations_used = 0

        best_goal_node: RRTNode | None = None
        best_goal_cost = float("inf")

        for i in range(self._max_iter):
            self._iterations_used = i + 1

            if self._rng.random() < 0.1:
                rand_point = goal
            else:
                rand_point = self._random_point()

            nearest = self._nearest(rand_point)
            new_pos = _steer(nearest.position, rand_point, self._step_size)

            if self._collision_fn(new_pos[0], new_pos[1], new_pos[2]):
                continue
            if not self._collision_free_path(nearest.position, new_pos):
                continue

            new_cost = nearest.cost + _dist(nearest.position, new_pos)
            new_node = RRTNode(position=new_pos, parent=nearest, cost=new_cost)

            if self._rewire_radius > 0:
                near = self._near_nodes(new_pos)
                for n in near:
                    candidate_cost = n.cost + _dist(n.position, new_pos)
                    if candidate_cost < new_node.cost and self._collision_free_path(n.position, new_pos):
                        new_node.parent = n
                        new_node.cost = candidate_cost

                for n in near:
                    rewire_cost = new_node.cost + _dist(new_node.position, n.position)
                    if rewire_cost < n.cost and self._collision_free_path(new_node.position, n.position):
                        if n.parent is not None:
                            n.parent.children = [c for c in n.parent.children if c is not n]
                        n.parent = new_node
                        n.cost = rewire_cost
                        new_node.children.append(n)

            if new_node.parent is not None:
                new_node.parent.children.append(new_node)
            self._nodes.append(new_node)

            if _dist(new_pos, goal) <= self._goal_radius:
                if new_node.cost < best_goal_cost:
                    best_goal_node = new_node
                    best_goal_cost = new_node.cost

        if best_goal_node is None:
            return []

        path: list[tuple[float, float, float]] = []
        node: RRTNode | None = best_goal_node
        while node is not None:
            path.append(node.position)
            node = node.parent
        path.reverse()
        return path

    def smooth_path(
        self, path: list[tuple[float, float, float]], iterations: int = 50
    ) -> list[tuple[float, float, float]]:
        """Post-process: shortcut random pairs to reduce path length."""
        if len(path) <= 2:
            return path
        smoothed = list(path)
        for _ in range(iterations):
            if len(smoothed) <= 2:
                break
            i = self._rng.randint(0, len(smoothed) - 2)
            j = self._rng.randint(i + 1, len(smoothed) - 1)
            if self._collision_free_path(smoothed[i], smoothed[j]):
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return smoothed

    def get_summary(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "iterations_used": self._iterations_used,
            "bounds_min": list(self._bounds_min),
            "bounds_max": list(self._bounds_max),
            "step_size": self._step_size,
            "goal_radius": self._goal_radius,
        }
