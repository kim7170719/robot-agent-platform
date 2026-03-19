"""
BasePlugin — abstract interface for all perception plugins.
Every plugin must implement start(), update(), stop() per Whitepaper 4.1.
Provides shared _active lifecycle management to reduce duplication.
"""

from abc import ABC, abstractmethod

from core.event_bus import EventBus


class BasePlugin(ABC):
    """
    Plugins read sensor data and publish events to the bus.
    They never import or call other modules directly.

    Subclasses only need to implement ``on_update()``; the
    start/stop active-flag lifecycle is handled here.
    """

    def __init__(self, event_bus: EventBus, plugin_id: str) -> None:
        self._bus = event_bus
        self._plugin_id = plugin_id
        self._active = False

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def update(self) -> None:
        if not self._active:
            return
        self.on_update()

    @abstractmethod
    def on_update(self) -> None:
        """Read sensor data and publish an event to the bus."""
