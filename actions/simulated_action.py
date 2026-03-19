"""
SimulatedAction — default ActionLayer that delegates to PhysicsWorld.

Extracts the movement logic that was previously hard-coded in
PhysicsAgentBase.act(), wrapping it behind the ActionLayer interface
so agents are world-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.physics_world import PhysicsWorld

from actions.base_action import ActionLayer


class SimulatedAction(ActionLayer):
    """
    Executes movement commands against a ``PhysicsWorld`` instance.

    Parameters
    ----------
    world : PhysicsWorld
        The physics simulation to act upon.
    robot_id : str | None
        Which robot in a multi-robot world to control.
    """

    def __init__(self, world: PhysicsWorld, robot_id: str | None = None) -> None:
        self._world = world
        self._robot_id = robot_id

    @property
    def world(self) -> PhysicsWorld:
        return self._world

    @property
    def robot_id(self) -> str | None:
        return self._robot_id

    def execute(self, command: dict) -> dict:
        direction = command.get("direction")
        if direction is None:
            return {"success": False, "reason": "missing direction"}

        pos_before = self._world.get_robot_position(self._robot_id)
        success = self._world.move_robot(direction, self._robot_id)
        pos_after = self._world.get_robot_position(self._robot_id)

        result: dict = {
            "success": success,
            "direction": direction,
            "position_before": list(pos_before),
            "new_position": list(pos_after),
        }
        if self._robot_id is not None:
            result["robot_id"] = self._robot_id
        return result
