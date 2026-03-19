"""
RealWorldAdapter — ActionLayer for real hardware control.

Provides a bridge between the platform's ActionLayer interface and
physical robot hardware via pluggable transport backends (serial,
gRPC, ROS, HTTP, etc.).

For safety, all commands pass through a CommandValidator that checks
joint limits, velocity bounds, and emergency stop conditions before
forwarding to the hardware transport.

When no transport is connected, operates in ``dry_run`` mode:
logs commands without executing, enabling offline testing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from actions.base_action import ActionLayer
from core.event_bus import EventBus, _make_event


class HardwareTransport(ABC):
    """
    Abstract transport layer — subclass this for your specific
    hardware protocol (serial, gRPC, ROS2 action client, etc.).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    def send_command(self, command: dict) -> dict:
        """Send a command and return the hardware response."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is active."""


class DryRunTransport(HardwareTransport):
    """No-op transport for offline testing. Logs commands only."""

    def __init__(self) -> None:
        self._connected = False
        self._command_log: list[dict] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def send_command(self, command: dict) -> dict:
        self._command_log.append(command)
        return {
            "success": True,
            "mode": "dry_run",
            "command": command,
            "timestamp": int(time.time() * 1000),
        }

    def is_connected(self) -> bool:
        return self._connected

    @property
    def command_log(self) -> list[dict]:
        return list(self._command_log)


class CommandValidator:
    """
    Validates commands before they reach hardware.
    Checks joint limits, velocity bounds, and e-stop.
    """

    def __init__(
        self,
        joint_limits: list[tuple[float, float]] | None = None,
        max_velocity: float = 2.0,
    ) -> None:
        self._joint_limits = joint_limits or []
        self._max_velocity = max_velocity
        self._e_stop = False
        self._rejected_count = 0

    @property
    def e_stop_active(self) -> bool:
        return self._e_stop

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def activate_e_stop(self) -> None:
        self._e_stop = True

    def release_e_stop(self) -> None:
        self._e_stop = False

    def validate(self, command: dict) -> tuple[bool, str]:
        if self._e_stop:
            self._rejected_count += 1
            return False, "emergency stop active"

        positions = command.get("joint_positions")
        if positions and self._joint_limits:
            for i, pos in enumerate(positions):
                if i < len(self._joint_limits):
                    low, high = self._joint_limits[i]
                    if pos < low or pos > high:
                        self._rejected_count += 1
                        return False, f"joint {i} out of limits: {pos} not in [{low}, {high}]"

        velocity = command.get("velocity")
        if velocity is not None and abs(velocity) > self._max_velocity:
            self._rejected_count += 1
            return False, f"velocity {velocity} exceeds max {self._max_velocity}"

        return True, "ok"


class RealWorldAdapter(ActionLayer):
    """
    Production-grade ActionLayer that routes commands through a
    validator and transport to real hardware.

    Parameters
    ----------
    transport : HardwareTransport
        The communication backend (serial, gRPC, etc.).
    event_bus : EventBus | None
        If provided, publishes ``action.hardware`` events for monitoring.
    validator : CommandValidator | None
        Optional command validator. If None, a permissive default is used.
    source : str
        Event source identifier.
    """

    def __init__(
        self,
        transport: HardwareTransport,
        event_bus: EventBus | None = None,
        validator: CommandValidator | None = None,
        source: str = "real_world_adapter",
    ) -> None:
        self._transport = transport
        self._bus = event_bus
        self._validator = validator or CommandValidator()
        self._source = source
        self._execution_count = 0

    @property
    def transport(self) -> HardwareTransport:
        return self._transport

    @property
    def validator(self) -> CommandValidator:
        return self._validator

    @property
    def execution_count(self) -> int:
        return self._execution_count

    def connect(self) -> bool:
        return self._transport.connect()

    def disconnect(self) -> None:
        self._transport.disconnect()

    def execute(self, command: dict) -> dict:
        valid, reason = self._validator.validate(command)
        if not valid:
            result = {"success": False, "reason": reason, "rejected": True}
            self._publish_event(command, result)
            return result

        if not self._transport.is_connected():
            result = {"success": False, "reason": "transport not connected"}
            self._publish_event(command, result)
            return result

        result = self._transport.send_command(command)
        self._execution_count += 1
        self._publish_event(command, result)
        return result

    def _publish_event(self, command: dict, result: dict) -> None:
        if self._bus is None:
            return
        self._bus.publish(_make_event("action.hardware", self._source, {
            "command": command,
            "result": result,
            "execution_index": self._execution_count,
        }))

    def get_status(self) -> dict:
        return {
            "connected": self._transport.is_connected(),
            "e_stop": self._validator.e_stop_active,
            "execution_count": self._execution_count,
            "rejected_count": self._validator.rejected_count,
        }
