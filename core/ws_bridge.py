"""
WebSocketBridge — a BasePlugin that subscribes to all EventBus events
(wildcard) and queues them for broadcast to WebSocket clients.

The bridge itself is synchronous and does NOT run an event loop.
It simply collects events; the dashboard server polls ``drain()``
to get queued events and sends them over the wire.
"""

from __future__ import annotations

import json
from collections import deque

from core.event_bus import EventBus, WILDCARD
from plugins.base_plugin import BasePlugin


_DEFAULT_MAX_QUEUE = 5000


class WebSocketBridge(BasePlugin):
    """
    Collects every EventBus event into an internal queue.

    The dashboard server calls ``drain()`` to retrieve and clear
    queued events, then broadcasts them to connected WebSocket clients.
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_queue: int = _DEFAULT_MAX_QUEUE,
    ) -> None:
        super().__init__(event_bus, plugin_id="ws_bridge")
        self._max_queue = max_queue
        self._queue: deque[dict] = deque(maxlen=max_queue)
        self._total_queued = 0
        self._connected_clients = 0

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def total_queued(self) -> int:
        return self._total_queued

    @property
    def connected_clients(self) -> int:
        return self._connected_clients

    @connected_clients.setter
    def connected_clients(self, value: int) -> None:
        self._connected_clients = max(0, value)

    def start(self) -> None:
        super().start()
        self._bus.subscribe(WILDCARD, self._on_event)

    def stop(self) -> None:
        self._bus.unsubscribe(WILDCARD, self._on_event)
        super().stop()

    def _on_event(self, event: dict) -> None:
        self._queue.append(event)
        self._total_queued += 1

    def on_update(self) -> None:
        pass

    def drain(self) -> list[dict]:
        """Return all queued events and clear the queue."""
        events = list(self._queue)
        self._queue.clear()
        return events

    def peek(self, n: int = 10) -> list[dict]:
        """Return the last *n* events without removing them."""
        items = list(self._queue)
        return items[-n:]

    def get_status(self) -> dict:
        return {
            "active": self.active,
            "queue_size": self.queue_size,
            "total_queued": self._total_queued,
            "max_queue": self._max_queue,
            "connected_clients": self._connected_clients,
        }

    @staticmethod
    def serialize_event(event: dict) -> str:
        """JSON-serialize an event for WebSocket transmission."""
        return json.dumps(event, default=str)
