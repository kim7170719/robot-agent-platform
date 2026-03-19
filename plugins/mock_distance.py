"""
MockDistance — simulated distance sensor plugin.
Reads distance readings from SensorSim and publishes them to the Event Bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.sensor_sim import SensorSim

from core.event_bus import EventBus
from plugins.base_plugin import BasePlugin


class MockDistance(BasePlugin):
    """Publishes ``sensor.distance`` events every update cycle."""

    def __init__(self, event_bus: EventBus, sensor_sim: SensorSim) -> None:
        super().__init__(event_bus, plugin_id="mock_distance")
        self._sensor_sim = sensor_sim

    def on_update(self) -> None:
        self._bus.publish(self._sensor_sim.read_distance())
