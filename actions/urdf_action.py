"""
URDFAction — ActionLayer for controlling URDF robot joints.

Commands use ``{"joint_positions": [float, ...]}`` to set target positions,
or ``{"direction": str}`` to map high-level directions to preset joint
configurations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.urdf_robot import URDFRobot

import pybullet as p

from actions.base_action import ActionLayer

_DIRECTION_PRESETS = {
    "forward":  [0.5, -0.3],
    "backward": [-0.5, 0.3],
    "up":       [-0.8, -0.8],
    "down":     [0.8, 0.8],
    "left":     [0.3, 0.0],
    "right":    [-0.3, 0.0],
}


class URDFAction(ActionLayer):
    """
    Translates high-level commands into joint-space control
    for a ``URDFRobot``.

    Supports two command formats:
      - ``{"joint_positions": [float, ...]}`` — direct joint control
      - ``{"direction": str}`` — mapped to preset joint targets
    """

    def __init__(
        self,
        urdf_robot: URDFRobot,
        physics_client: int,
        sim_steps: int = 120,
    ) -> None:
        self._robot = urdf_robot
        self._client = physics_client
        self._sim_steps = sim_steps

    @property
    def urdf_robot(self) -> URDFRobot:
        return self._robot

    def execute(self, command: dict) -> dict:
        target_position = command.get("target_position")
        joint_positions = command.get("joint_positions")
        direction = command.get("direction")

        if target_position is not None:
            return self._execute_ik(target_position)
        elif joint_positions is not None:
            return self._execute_joints(joint_positions)
        elif direction is not None:
            preset = _DIRECTION_PRESETS.get(direction)
            if preset is None:
                return {"success": False, "reason": f"Unknown direction: {direction}"}
            return self._execute_joints(preset)
        else:
            return {"success": False, "reason": "No target_position, joint_positions, or direction"}

    def _execute_ik(self, target: list[float]) -> dict:
        return self._robot.move_end_effector_to(
            target[0], target[1], target[2],
            sim_steps=self._sim_steps,
        )

    def _execute_joints(self, positions: list[float]) -> dict:
        ee_before = self._robot.get_end_effector_position()
        self._robot.set_joint_positions(positions)

        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        ee_after = self._robot.get_end_effector_position()
        actual = self._robot.get_joint_positions()

        return {
            "success": True,
            "target_positions": positions[:self._robot.num_joints],
            "actual_positions": actual,
            "ee_before": list(ee_before),
            "ee_after": list(ee_after),
            "ee_distance": round(math.dist(ee_before, ee_after), 4),
        }
