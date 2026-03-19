"""
GraspPlanner — pick-and-place pipeline for URDF robot arms.

Given a target object position and the arm's IK solver, computes
a sequence of grasp poses (approach, grasp, lift, place) and
returns the full action sequence for execution via URDFAction.

Works with the existing URDFRobot.solve_ik() and ActionLayer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.urdf_robot import URDFRobot


@dataclass
class GraspPose:
    """A pose in the grasp sequence."""
    phase: str
    position: tuple[float, float, float]
    joint_targets: list[float] | None = None
    ik_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "position": list(self.position),
            "joint_targets": self.joint_targets,
            "ik_error": self.ik_error,
        }


class GraspPlanner:
    """
    Plans a pick-and-place trajectory:
      1. **approach** — move above the object
      2. **grasp**    — descend to the object
      3. **lift**     — lift the object
      4. **transport** — move above the place position
      5. **place**    — lower to the place position
      6. **retreat**  — move up and away

    Parameters
    ----------
    robot : URDFRobot
        The arm with IK capability.
    approach_height : float
        Height above target for approach/retreat.
    lift_height : float
        Height above grasp point for lifting.
    """

    def __init__(
        self,
        robot: URDFRobot,
        approach_height: float = 0.15,
        lift_height: float = 0.10,
    ) -> None:
        self._robot = robot
        self._approach_h = approach_height
        self._lift_h = lift_height
        self._last_plan: list[GraspPose] = []

    @property
    def last_plan(self) -> list[GraspPose]:
        return list(self._last_plan)

    def plan_pick(
        self, target: tuple[float, float, float]
    ) -> list[GraspPose]:
        """Plan the pick (grasp) phase only."""
        x, y, z = target
        poses = [
            self._solve("approach", (x, y, z + self._approach_h)),
            self._solve("grasp", (x, y, z)),
            self._solve("lift", (x, y, z + self._lift_h)),
        ]
        return poses

    def plan_place(
        self, place_target: tuple[float, float, float]
    ) -> list[GraspPose]:
        """Plan the place phase only."""
        x, y, z = place_target
        poses = [
            self._solve("transport", (x, y, z + self._approach_h)),
            self._solve("place", (x, y, z)),
            self._solve("retreat", (x, y, z + self._approach_h)),
        ]
        return poses

    def plan_pick_and_place(
        self,
        pick_target: tuple[float, float, float],
        place_target: tuple[float, float, float],
    ) -> list[GraspPose]:
        """Full pick-and-place trajectory."""
        pick_poses = self.plan_pick(pick_target)
        place_poses = self.plan_place(place_target)
        full = pick_poses + place_poses
        self._last_plan = full
        return full

    def _solve(self, phase: str, pos: tuple[float, float, float]) -> GraspPose:
        joints = self._robot.solve_ik(pos)
        ee = self._estimate_ee(joints)
        error = math.dist(ee, pos)
        return GraspPose(
            phase=phase,
            position=pos,
            joint_targets=joints,
            ik_error=round(error, 4),
        )

    def _estimate_ee(self, joints: list[float]) -> tuple[float, float, float]:
        """
        Estimate end-effector position for given joint angles without
        actually moving the robot (uses FK via current robot state
        as approximation).
        """
        return self._robot.get_end_effector_position()

    def get_action_commands(
        self, plan: list[GraspPose] | None = None
    ) -> list[dict]:
        """
        Convert a grasp plan into a list of ActionLayer commands
        (for use with URDFAction).
        """
        plan = plan or self._last_plan
        commands = []
        for pose in plan:
            if pose.joint_targets is not None:
                commands.append({
                    "joint_positions": pose.joint_targets,
                    "phase": pose.phase,
                })
            else:
                commands.append({
                    "target_position": list(pose.position),
                    "phase": pose.phase,
                })
        return commands

    def evaluate_plan(self, plan: list[GraspPose] | None = None) -> dict:
        """Evaluate plan quality."""
        plan = plan or self._last_plan
        if not plan:
            return {"phases": 0, "total_ik_error": 0, "max_ik_error": 0, "feasible": False}
        errors = [p.ik_error for p in plan]
        return {
            "phases": len(plan),
            "total_ik_error": round(sum(errors), 4),
            "max_ik_error": round(max(errors), 4),
            "avg_ik_error": round(sum(errors) / len(errors), 4),
            "feasible": all(p.joint_targets is not None for p in plan),
            "phase_names": [p.phase for p in plan],
        }
