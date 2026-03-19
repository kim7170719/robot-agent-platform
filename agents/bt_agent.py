"""
BTAgent — Behavior Tree driven agent for composable multi-phase missions.

Instead of a fixed perceive→decide→act pipeline, the BTAgent ticks
a configurable behavior tree each engine step. The tree can mix
conditions (sensor checks), actions (movement, IK), and control
flow (sequence, selector, parallel) for complex mission logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.bt_nodes import BTNode

from agents.base_agent import BaseAgent
from agents.bt_nodes import Status
from core.event_bus import EventBus, _make_event


class BTAgent(BaseAgent):
    """
    Behavior Tree agent. Each engine step ticks the root BT node.

    Parameters
    ----------
    event_bus : EventBus
    root : BTNode
        The root of the behavior tree to execute.
    agent_id : str
    max_ticks_per_step : int
        How many BT ticks per engine step (default 1).
    """

    def __init__(
        self,
        event_bus: EventBus,
        root: BTNode,
        agent_id: str = "bt_agent",
        max_ticks_per_step: int = 1,
    ) -> None:
        super().__init__(event_bus, agent_id=agent_id)
        self._root = root
        self._max_ticks = max(1, max_ticks_per_step)
        self._tick_count = 0
        self._last_status: Status = Status.FAILURE
        self._success_count = 0
        self._failure_count = 0

    @property
    def root(self) -> BTNode:
        return self._root

    @root.setter
    def root(self, node: BTNode) -> None:
        self._root = node

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def last_status(self) -> Status:
        return self._last_status

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def perceive(self) -> None:
        pass

    def decide(self) -> None:
        for _ in range(self._max_ticks):
            self._last_status = self._root.tick()
            self._tick_count += 1

            if self._last_status == Status.SUCCESS:
                self._success_count += 1
            elif self._last_status == Status.FAILURE:
                self._failure_count += 1

    def act(self) -> None:
        self._bus.publish(_make_event("bt.tick", self._agent_id, {
            "tick": self._tick_count,
            "status": self._last_status.name,
            "success_total": self._success_count,
            "failure_total": self._failure_count,
        }))

    def reset_tree(self) -> None:
        """Reset the BT for a new mission."""
        self._root.reset()
        self._tick_count = 0
        self._last_status = Status.FAILURE
        self._success_count = 0
        self._failure_count = 0

    def get_summary(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "tick_count": self._tick_count,
            "last_status": self._last_status.name,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
        }
