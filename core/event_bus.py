"""
Pub/sub Event Bus — the central communication backbone of IR-AAP.
All modules publish and subscribe through this bus; no direct coupling.
"""

import time
import warnings
from collections import defaultdict, deque
from typing import Callable


REQUIRED_EVENT_FIELDS = {"timestamp", "type", "source", "data"}

WILDCARD = "*"


def _make_event(event_type: str, source: str, data: dict) -> dict:
    """Factory helper to build a protocol-compliant event dict."""
    return {
        "timestamp": int(time.time() * 1000),
        "type": event_type,
        "source": source,
        "data": data,
    }


def _validate_event(event: dict) -> None:
    if not isinstance(event, dict):
        raise TypeError(f"Event must be a dict, got {type(event).__name__}")
    missing = REQUIRED_EVENT_FIELDS - event.keys()
    if missing:
        raise ValueError(f"Event missing required fields: {missing}")


class EventBus:
    """
    Thread-unsafe, deterministic pub/sub bus.

    Features:
      - Wildcard ("*") subscribers receive every event.
      - Bounded history (max_history) prevents memory leaks in long runs.
        Set max_history=0 to disable history recording entirely.
      - Type-indexed history for O(1) lookups by event type.
      - Callback exception isolation: a failing subscriber never blocks others.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        if max_history < 0:
            raise ValueError("max_history must be >= 0")
        self._subscribers: dict[str, list[Callable[[dict], None]]] = {}
        self._max_history = max_history
        self._history_enabled = max_history != 0
        self._history: deque[dict] = deque(maxlen=max_history if max_history > 0 else None)
        self._type_index: dict[str, list[dict]] = defaultdict(list)
        self._publish_count = 0

    @property
    def max_history(self) -> int:
        return self._max_history

    @property
    def publish_count(self) -> int:
        return self._publish_count

    def subscribe(self, event_type: str, callback: Callable[[dict], None]) -> None:
        """Subscribe *callback* to *event_type*. Use ``"*"`` for all events."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[dict], None]) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]

    def publish(self, event: dict) -> None:
        _validate_event(event)
        self._publish_count += 1

        if self._history_enabled:
            if self._max_history > 0 and len(self._history) >= self._max_history:
                evicted = self._history[0]
                evicted_list = self._type_index.get(evicted["type"])
                if evicted_list:
                    try:
                        evicted_list.remove(evicted)
                    except ValueError:
                        pass
            self._history.append(event)
            self._type_index[event["type"]].append(event)

        self._dispatch(event["type"], event)

        if event["type"] != WILDCARD:
            self._dispatch(WILDCARD, event)

    def _dispatch(self, event_type: str, event: dict) -> None:
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(event)
            except Exception as exc:
                warnings.warn(
                    f"EventBus: subscriber {cb!r} raised {exc!r} for event type "
                    f"'{event['type']}'; skipping.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def get_history(self, event_type: str | None = None) -> list[dict]:
        if event_type is None:
            return list(self._history)
        return list(self._type_index.get(event_type, []))

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()
        self._type_index.clear()
        self._publish_count = 0
