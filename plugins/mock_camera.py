"""
MockCamera — simulated camera sensor plugin.
Reads a camera frame from SensorSim and publishes it to the Event Bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.sensor_sim import SensorSim

from core.event_bus import EventBus
from plugins.base_plugin import BasePlugin


class MockCamera(BasePlugin):
    """Publishes ``sensor.camera`` events every update cycle."""

    def __init__(self, event_bus: EventBus, sensor_sim: SensorSim) -> None:
        super().__init__(event_bus, plugin_id="mock_camera")
        self._sensor_sim = sensor_sim

    def on_update(self) -> None:
        self._bus.publish(self._sensor_sim.read_camera())
