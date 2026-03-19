"""
JointSensor — plugin that reads joint states from a URDFRobot
and publishes ``sensor.joints`` events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.urdf_robot import URDFRobot

from core.event_bus import EventBus, _make_event
from plugins.base_plugin import BasePlugin


class JointSensor(BasePlugin):
    """
    Reads joint positions, velocities, and end-effector position
    from a ``URDFRobot`` and publishes ``sensor.joints`` events.
    """

    def __init__(self, event_bus: EventBus, urdf_robot: URDFRobot) -> None:
        super().__init__(event_bus, plugin_id="joint_sensor")
        self._robot = urdf_robot

    @property
    def urdf_robot(self) -> URDFRobot:
        return self._robot

    def on_update(self) -> None:
        data = {
            "robot_id": self._robot.robot_id,
            "joint_positions": self._robot.get_joint_positions(),
            "joint_velocities": self._robot.get_joint_velocities(),
            "end_effector_position": list(self._robot.get_end_effector_position()),
            "num_joints": self._robot.num_joints,
        }
        self._bus.publish(_make_event("sensor.joints", self._plugin_id, data))
