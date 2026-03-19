"""
TaskAllocator — multi-robot cooperative task assignment plugin.

Publishes ``task.assigned`` events, listens for ``task.completed`` events.
Uses a simple nearest-idle-robot strategy to assign waypoint tasks to
robots in a multi-robot PhysicsWorld.

Per Whitepaper §6: the execution model is Collect → Feed → Decide → Execute.
The allocator sits on the EventBus and coordinates without direct coupling.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin


class Task:
    """A single waypoint task for a robot."""

    __slots__ = ("task_id", "target", "assigned_to", "status")

    def __init__(self, task_id: str, target: tuple[float, float, float]) -> None:
        self.task_id = task_id
        self.target = target
        self.assigned_to: str | None = None
        self.status: str = "pending"  # pending | assigned | completed

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "target": list(self.target),
            "assigned_to": self.assigned_to,
            "status": self.status,
        }


class TaskAllocator(BasePlugin):
    """
    Manages a queue of waypoint tasks and assigns them to the
    nearest idle robot. Publishes ``goal.target`` events (consumed
    by GoalAgent / AStarAgent) and ``task.assigned`` events for
    monitoring.

    Listens to ``goal.reached`` to mark tasks complete and re-assign.
    """

    def __init__(self, event_bus: EventBus, world: PhysicsWorld) -> None:
        super().__init__(event_bus, plugin_id="task_allocator")
        self._world = world
        self._tasks: list[Task] = []
        self._busy_robots: set[str] = set()
        self._completed_count = 0
        self._bus.subscribe("goal.reached", self._on_goal_reached)

    @property
    def tasks(self) -> list[dict]:
        return [t.to_dict() for t in self._tasks]

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._tasks if t.status == "pending")

    @property
    def completed_count(self) -> int:
        return self._completed_count

    @property
    def busy_robots(self) -> set[str]:
        return set(self._busy_robots)

    def add_task(self, task_id: str, x: float, y: float, z: float = 0.31) -> None:
        self._tasks.append(Task(task_id, (x, y, z)))

    def add_tasks(self, tasks: list[dict]) -> int:
        """Batch add: each dict has ``task_id``, ``x``, ``y``, optional ``z``."""
        count = 0
        for t in tasks:
            self.add_task(t["task_id"], t["x"], t["y"], t.get("z", 0.31))
            count += 1
        return count

    def _idle_robots(self) -> list[str]:
        return [r for r in self._world.get_robot_ids() if r not in self._busy_robots]

    def _nearest_robot(self, target: tuple[float, float, float], candidates: list[str]) -> str | None:
        if not candidates:
            return None
        best, best_dist = None, float("inf")
        for rid in candidates:
            pos = self._world.get_robot_position(rid)
            d = math.dist(pos, target)
            if d < best_dist:
                best, best_dist = rid, d
        return best

    def _assign_next(self) -> bool:
        """Try to assign one pending task. Returns True if assigned."""
        idle = self._idle_robots()
        if not idle:
            return False

        for task in self._tasks:
            if task.status != "pending":
                continue
            robot = self._nearest_robot(task.target, idle)
            if robot is None:
                return False

            task.assigned_to = robot
            task.status = "assigned"
            self._busy_robots.add(robot)

            self._bus.publish(_make_event("task.assigned", self._plugin_id, {
                "task_id": task.task_id,
                "robot_id": robot,
                "target": list(task.target),
            }))
            self._bus.publish(_make_event("goal.target", self._plugin_id, {
                "target": list(task.target),
                "robot_id": robot,
                "reached": False,
            }))
            return True

        return False

    def _on_goal_reached(self, event: dict) -> None:
        for task in self._tasks:
            if task.status == "assigned" and task.assigned_to:
                task.status = "completed"
                self._busy_robots.discard(task.assigned_to)
                self._completed_count += 1
                self._bus.publish(_make_event("task.completed", self._plugin_id, {
                    "task_id": task.task_id,
                    "robot_id": task.assigned_to,
                }))
                break

    def on_update(self) -> None:
        while self._assign_next():
            pass

    def stop(self) -> None:
        self._bus.unsubscribe("goal.reached", self._on_goal_reached)
        super().stop()

    def get_summary(self) -> dict:
        return {
            "total_tasks": len(self._tasks),
            "pending": self.pending_count,
            "assigned": sum(1 for t in self._tasks if t.status == "assigned"),
            "completed": self._completed_count,
            "busy_robots": list(self._busy_robots),
        }
