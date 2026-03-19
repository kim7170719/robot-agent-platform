"""
AStarAgent — 3D goal-directed agent using A* global path planning.
Builds an occupancy grid from sensor data, plans an optimal path,
then follows waypoints while avoiding obstacles reactively.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld
    from actions.base_action import ActionLayer

from agents.physics_agent_base import PhysicsAgentBase
from core.event_bus import EventBus, _make_event
from simulator.physics_world import DIRECTIONS_3D

_GOAL_REACHED_RADIUS = 0.8
_GRID_RESOLUTION = 1.0


def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def astar_search(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    blocked: set[tuple[int, int, int]],
    max_iterations: int = 5000,
) -> list[tuple[int, int, int]]:
    """
    Pure A* on a 3D integer grid. Returns path from start to goal
    (inclusive), or empty list if unreachable within max_iterations.
    """
    open_set: list[tuple[float, tuple[int, int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    g_score: dict[tuple[int, int, int], float] = defaultdict(lambda: float("inf"))
    g_score[start] = 0.0
    iterations = 0

    neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    while open_set and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            node: tuple[int, int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for dx, dy, dz in neighbors_6:
            nb = (current[0] + dx, current[1] + dy, current[2] + dz)
            if nb in blocked:
                continue
            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + _heuristic(nb, goal)
                came_from[nb] = current
                heapq.heappush(open_set, (f, nb))

    return []


def _direction_from_delta(dx: int, dy: int, dz: int) -> str | None:
    """Map a grid step delta to a named direction."""
    for name, (ddx, ddy, ddz) in DIRECTIONS_3D.items():
        if (dx, dy, dz) == (int(ddx), int(ddy), int(ddz)):
            return name
    return None


class AStarAgent(PhysicsAgentBase):
    """
    Plans a global A* path then follows it step-by-step.
    Re-plans when the path is blocked or a new goal arrives.
    """

    def __init__(
        self,
        event_bus: EventBus,
        world: PhysicsWorld,
        robot_id: str | None = None,
        action_layer: ActionLayer | None = None,
    ) -> None:
        super().__init__(
            event_bus, world, agent_id="astar_agent",
            robot_id=robot_id, action_layer=action_layer,
        )
        self._goal: tuple[float, float, float] | None = None
        self._goal_reached = False
        self._path: list[tuple[int, int, int]] = []
        self._blocked_cells: set[tuple[int, int, int]] = set()
        self._replan_needed = True
        self._bus.subscribe("goal.target", self._on_goal)

    @property
    def goal(self) -> tuple[float, float, float] | None:
        return self._goal

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def path(self) -> list[tuple[int, int, int]]:
        return list(self._path)

    @property
    def blocked_cells(self) -> set[tuple[int, int, int]]:
        return set(self._blocked_cells)

    def cleanup(self) -> None:
        super().cleanup()
        self._bus.unsubscribe("goal.target", self._on_goal)

    def _on_goal(self, event: dict) -> None:
        target_data = event.get("data", {}).get("target")
        if target_data and len(target_data) >= 3:
            self._goal = (target_data[0], target_data[1], target_data[2])
            self._goal_reached = False
            self._replan_needed = True

    def _on_distance(self, event: dict) -> None:
        src = event.get("data", {}).get("robot_id")
        if src is not None and self._robot_id is not None and src != self._robot_id:
            return
        self._latest_distance = event

        distances = event.get("data", {}).get("distances", {})
        pos = self._world.get_robot_position(self._robot_id)
        for name, (dx, dy, dz) in DIRECTIONS_3D.items():
            info = distances.get(name, {})
            if info.get("blocked", False) and info.get("distance", 999) < 2.0:
                cell = (round(pos[0] + dx), round(pos[1] + dy), round(pos[2] + dz))
                if cell not in self._blocked_cells:
                    self._blocked_cells.add(cell)
                    self._replan_needed = True

    def _plan(self) -> None:
        if self._goal is None:
            self._path = []
            return
        start = self._quantize_pos()
        goal = (round(self._goal[0]), round(self._goal[1]), round(self._goal[2]))
        self._path = astar_search(start, goal, self._blocked_cells)
        if self._path and self._path[0] == start:
            self._path.pop(0)
        self._replan_needed = False

    def decide(self) -> None:
        if self._latest_distance is None:
            self._chosen_action = "forward"
            return

        if self._goal is not None:
            pos = self._world.get_robot_position(self._robot_id)
            if math.dist(pos, self._goal) < _GOAL_REACHED_RADIUS:
                self._goal_reached = True
                self._chosen_action = None
                self._bus.publish(_make_event("goal.reached", self._agent_id, {
                    "goal": list(self._goal),
                    "position": list(pos),
                }))
                return

        if self._replan_needed:
            self._plan()

        if self._path:
            current = self._quantize_pos()
            next_cell = self._path[0]
            dx = next_cell[0] - current[0]
            dy = next_cell[1] - current[1]
            dz = next_cell[2] - current[2]
            direction = _direction_from_delta(dx, dy, dz)
            if direction is not None:
                self._chosen_action = direction
                self._path.pop(0)
            else:
                self._replan_needed = True
                self._chosen_action = None
        else:
            candidates = self._safe_candidates()
            if candidates:
                candidates.sort(key=lambda c: (c[1], c[2]))
                self._chosen_action = candidates[0][0]
            else:
                self._chosen_action = None

    def _build_move_event_data(self, pos_before: tuple) -> dict:
        data = super()._build_move_event_data(pos_before)
        data["goal"] = list(self._goal) if self._goal else None
        data["path_remaining"] = len(self._path)
        data["blocked_cells_count"] = len(self._blocked_cells)
        if self._goal:
            data["goal_distance"] = round(
                math.dist(self._world.get_robot_position(self._robot_id), self._goal), 3
            )
        return data
