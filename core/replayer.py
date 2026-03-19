"""
Replayer — re-publishes recorded JSONL events to an EventBus.
Supports real-time paced replay, fast-forward, filtered replay,
and step-by-step mode for offline debugging and regression testing.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from core.event_bus import EventBus


class Replayer:
    """
    Load a Recorder JSONL file and replay events through any EventBus.

    Modes:
      - ``replay(speed=1.0)`` — real-time paced (respects timestamp gaps)
      - ``replay(speed=0)`` — instant (fire all events as fast as possible)
      - ``step()`` — advance one event at a time
      - ``replay(event_filter=...)`` — only replay matching event types
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._events: list[dict] = []
        self._cursor: int = 0
        self._replaying = False
        self._replayed_count = 0

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def replayed_count(self) -> int:
        return self._replayed_count

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._events)

    def load(self, path: str) -> int:
        """Load events from a JSONL file. Returns event count."""
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        self._events = events
        self._cursor = 0
        self._replayed_count = 0
        return len(events)

    def load_events(self, events: list[dict]) -> int:
        """Load events from an in-memory list. Returns event count."""
        self._events = list(events)
        self._cursor = 0
        self._replayed_count = 0
        return len(events)

    def reset(self) -> None:
        """Reset cursor to the beginning without clearing events."""
        self._cursor = 0
        self._replayed_count = 0

    def step(self) -> dict | None:
        """Publish the next event and advance cursor. Returns the event or None."""
        if self._cursor >= len(self._events):
            return None
        event = self._events[self._cursor]
        self._bus.publish(event)
        self._cursor += 1
        self._replayed_count += 1
        return event

    def replay(
        self,
        speed: float = 0.0,
        event_filter: str | None = None,
        on_event: Callable[[dict, int], None] | None = None,
        max_events: int | None = None,
    ) -> int:
        """
        Replay loaded events through the bus.

        Args:
            speed: Playback multiplier. 0 = instant, 1.0 = real-time,
                   2.0 = double speed. Negative values treated as 0.
            event_filter: If set, only replay events matching this type.
            on_event: Optional callback ``(event, index) -> None`` after each publish.
            max_events: Stop after this many events (None = all).

        Returns:
            Number of events replayed.
        """
        self._replaying = True
        speed = max(0.0, speed)
        count = 0
        prev_ts: int | None = None

        while self._cursor < len(self._events) and self._replaying:
            if max_events is not None and count >= max_events:
                break

            event = self._events[self._cursor]

            if event_filter is not None and event.get("type") != event_filter:
                self._cursor += 1
                continue

            if speed > 0 and prev_ts is not None:
                ts = event.get("timestamp", 0)
                gap_ms = ts - prev_ts
                if gap_ms > 0:
                    time.sleep((gap_ms / 1000.0) / speed)

            prev_ts = event.get("timestamp", 0)
            self._bus.publish(event)
            count += 1
            self._replayed_count += 1

            if on_event is not None:
                on_event(event, self._cursor)

            self._cursor += 1

        self._replaying = False
        return count

    def stop(self) -> None:
        """Interrupt an ongoing replay."""
        self._replaying = False

    def get_remaining(self) -> list[dict]:
        """Return events that haven't been replayed yet."""
        return list(self._events[self._cursor:])

    def get_events_by_type(self, event_type: str) -> list[dict]:
        """Return all loaded events matching the given type."""
        return [e for e in self._events if e.get("type") == event_type]

    def get_summary(self) -> dict:
        """Summary of loaded events similar to Recorder.get_summary."""
        type_counts: dict[str, int] = {}
        for e in self._events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        ts_values = [e.get("timestamp", 0) for e in self._events]
        duration_ms = (max(ts_values) - min(ts_values)) if ts_values else 0
        return {
            "total_events": len(self._events),
            "replayed": self._replayed_count,
            "remaining": len(self._events) - self._cursor,
            "event_types": type_counts,
            "duration_ms": duration_ms,
        }
