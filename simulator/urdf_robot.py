"""
URDFRobot — loads and controls articulated URDF robot models in PyBullet.

Coexists with sphere robots in the same PhysicsWorld.  Provides joint-level
control (position, velocity, torque) and end-effector queries.

Ships with a bundled simple box-arm URDF for testing without external files.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import pybullet as p

if TYPE_CHECKING:
    pass

_BUNDLED_URDF = """\
<?xml version="1.0"?>
<robot name="box_arm">
  <link name="base_link">
    <visual><geometry><box size="0.2 0.2 0.1"/></geometry>
      <material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
    </visual>
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
    <inertial><mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <link name="link1">
    <visual><geometry><box size="0.05 0.05 0.3"/></geometry>
      <origin xyz="0 0 0.15"/>
      <material name="blue"><color rgba="0.2 0.4 0.9 1"/></material>
    </visual>
    <collision><geometry><box size="0.05 0.05 0.3"/></geometry>
      <origin xyz="0 0 0.15"/>
    </collision>
    <inertial><mass value="0.5"/>
      <origin xyz="0 0 0.15"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>

  <link name="link2">
    <visual><geometry><box size="0.04 0.04 0.25"/></geometry>
      <origin xyz="0 0 0.125"/>
      <material name="cyan"><color rgba="0.1 0.8 0.9 1"/></material>
    </visual>
    <collision><geometry><box size="0.04 0.04 0.25"/></geometry>
      <origin xyz="0 0 0.125"/>
    </collision>
    <inertial><mass value="0.3"/>
      <origin xyz="0 0 0.125"/>
      <inertia ixx="0.0005" iyy="0.0005" izz="0.00005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="10" velocity="2"/>
  </joint>

  <link name="gripper_link">
    <visual><geometry><box size="0.05 0.04 0.02"/></geometry>
      <origin xyz="0.025 0 0"/>
      <material name="orange"><color rgba="1.0 0.45 0.1 1"/></material>
    </visual>
    <collision><geometry><box size="0.05 0.04 0.02"/></geometry>
      <origin xyz="0.025 0 0"/>
    </collision>
    <inertial><mass value="0.04"/>
      <origin xyz="0.025 0 0"/>
      <inertia ixx="0.00001" iyy="0.00001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="gripper_slide" type="prismatic">
    <parent link="link2"/>
    <child link="gripper_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.08" effort="80" velocity="0.15"/>
  </joint>
</robot>
"""


class URDFRobot:
    """
    Loads a URDF model into an existing PyBullet physics client.

    Parameters
    ----------
    physics_client : int
        PyBullet client ID (from ``PhysicsWorld._client``).
    urdf_path : str | None
        Path to a .urdf file.  If ``None``, uses the bundled box-arm.
    position : tuple
        Base spawn position ``(x, y, z)``.
    robot_id : str
        Logical name for this URDF robot.
    """

    def __init__(
        self,
        physics_client: int,
        urdf_path: str | None = None,
        position: tuple[float, float, float] = (0, 0, 0),
        robot_id: str = "urdf_robot",
    ) -> None:
        self._client = physics_client
        self._robot_id = robot_id
        self._temp_file: str | None = None

        if urdf_path is None:
            bundle_dir = os.path.join(os.path.dirname(__file__), "_urdf_cache")
            os.makedirs(bundle_dir, exist_ok=True)
            path = os.path.join(bundle_dir, "box_arm_with_gripper.urdf")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(_BUNDLED_URDF)
            urdf_path = path

        self._body_id = p.loadURDF(
            urdf_path,
            basePosition=list(position),
            useFixedBase=True,
            physicsClientId=self._client,
        )

        self._num_joints = p.getNumJoints(self._body_id, physicsClientId=self._client)
        self._joint_info: list[dict] = []
        for i in range(self._num_joints):
            info = p.getJointInfo(self._body_id, i, physicsClientId=self._client)
            self._joint_info.append({
                "index": i,
                "name": info[1].decode("utf-8"),
                "type": info[2],
                "lower_limit": info[8],
                "upper_limit": info[9],
                "max_force": info[10],
                "max_velocity": info[11],
            })

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def joint_info(self) -> list[dict]:
        return list(self._joint_info)

    def get_joint_positions(self) -> list[float]:
        positions = []
        for i in range(self._num_joints):
            state = p.getJointState(self._body_id, i, physicsClientId=self._client)
            positions.append(state[0])
        return positions

    def get_joint_velocities(self) -> list[float]:
        velocities = []
        for i in range(self._num_joints):
            state = p.getJointState(self._body_id, i, physicsClientId=self._client)
            velocities.append(state[1])
        return velocities

    def get_joint_torques(self) -> list[float]:
        torques = []
        for i in range(self._num_joints):
            state = p.getJointState(self._body_id, i, physicsClientId=self._client)
            torques.append(state[3])
        return torques

    def set_gripper_openness(self, openness: float) -> None:
        """
        Set prismatic *gripper_slide* joint: ``0`` ≈ closed, ``1`` ≈ fully open.

        No physical grasp constraint — demo only; combine with contact plugins for tests.
        """
        openness = max(0.0, min(1.0, float(openness)))
        grip_idx: int | None = None
        for info in self._joint_info:
            if info["name"] == "gripper_slide":
                grip_idx = info["index"]
                break
        if grip_idx is None:
            return
        lo = float(self._joint_info[grip_idx]["lower_limit"])
        hi = float(self._joint_info[grip_idx]["upper_limit"])
        target = lo + openness * (hi - lo)
        info = self._joint_info[grip_idx]
        p.setJointMotorControl2(
            self._body_id,
            grip_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=info["max_force"],
            physicsClientId=self._client,
        )

    def set_joint_positions(self, positions: list[float]) -> None:
        """Set target joint positions via position control."""
        for i, pos in enumerate(positions[:self._num_joints]):
            info = self._joint_info[i]
            clamped = max(info["lower_limit"], min(info["upper_limit"], pos))
            p.setJointMotorControl2(
                self._body_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=clamped,
                force=info["max_force"],
                physicsClientId=self._client,
            )

    def reset_joints(self) -> None:
        """Reset all joints to zero position."""
        for i in range(self._num_joints):
            p.resetJointState(self._body_id, i, targetValue=0.0,
                              physicsClientId=self._client)

    def get_base_position(self) -> tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._client,
        )
        return (round(pos[0], 4), round(pos[1], 4), round(pos[2], 4))

    def get_end_effector_position(self) -> tuple[float, float, float]:
        """Position of the last link (end effector)."""
        if self._num_joints == 0:
            return self.get_base_position()
        state = p.getLinkState(
            self._body_id, self._num_joints - 1,
            physicsClientId=self._client,
        )
        pos = state[0]
        return (round(pos[0], 4), round(pos[1], 4), round(pos[2], 4))

    def solve_ik(
        self,
        target_position: tuple[float, float, float],
        max_iterations: int = 100,
        residual_threshold: float = 1e-4,
    ) -> list[float]:
        """
        Compute joint angles that place the end-effector at *target_position*
        using PyBullet's built-in damped least-squares IK solver.

        Returns the solved joint positions (list of floats).
        """
        end_effector_index = self._num_joints - 1

        lower = [info["lower_limit"] for info in self._joint_info]
        upper = [info["upper_limit"] for info in self._joint_info]
        ranges = [hi - lo for lo, hi in zip(lower, upper)]
        rest = [0.0] * self._num_joints

        joint_positions = p.calculateInverseKinematics(
            self._body_id,
            end_effector_index,
            list(target_position),
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest,
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold,
            physicsClientId=self._client,
        )
        out: list[float] = []
        for i, val in enumerate(joint_positions[:self._num_joints]):
            lo = float(self._joint_info[i]["lower_limit"])
            hi = float(self._joint_info[i]["upper_limit"])
            out.append(max(lo, min(hi, float(val))))
        return out

    def move_end_effector_to(
        self,
        x: float, y: float, z: float,
        sim_steps: int = 240,
        max_ik_iterations: int = 100,
    ) -> dict:
        """
        High-level IK command: move end-effector to (x, y, z).

        Solves IK, applies position control, steps the simulation,
        and returns a result dict with achieved position and error.
        """
        target = (x, y, z)
        ee_before = self.get_end_effector_position()
        joint_targets = self.solve_ik(target, max_iterations=max_ik_iterations)
        self.set_joint_positions(joint_targets)

        for _ in range(sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        ee_after = self.get_end_effector_position()
        error = math.dist(ee_after, target)

        return {
            "success": error < 0.1,
            "target": list(target),
            "achieved": list(ee_after),
            "error": round(error, 4),
            "joint_targets": joint_targets,
            "joint_actual": self.get_joint_positions(),
            "ee_before": list(ee_before),
            "ee_after": list(ee_after),
        }

    def get_state(self) -> dict:
        """Full robot state snapshot."""
        return {
            "robot_id": self._robot_id,
            "base_position": list(self.get_base_position()),
            "end_effector_position": list(self.get_end_effector_position()),
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "num_joints": self._num_joints,
        }

    def cleanup(self) -> None:
        """Remove temp URDF file if created."""
        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
            self._temp_file = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
