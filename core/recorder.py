"""
Recorder — records EventBus events to a JSON file for offline replay and analysis.
Subscribes via wildcard so it captures every event without coupling.
"""

from __future__ import annotations

import json
import time

from core.event_bus import EventBus, WILDCARD


class Recorder:
    """
    Subscribes to all events via WILDCARD and buffers them in memory.
    Call ``start()`` to begin recording, ``stop()`` to stop capturing.
    Call ``save()`` to write buffered events to a JSON Lines file.

    The output file is JSON Lines (one JSON object per line) for
    streaming-friendly reads and easy append behavior.
    """

    def __init__(self, event_bus: EventBus, output_path: str = "recording.jsonl") -> None:
        self._bus = event_bus
        self._path = output_path
        self._events: list[dict] = []
        self._recording = False
        self._start_time: float = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def event_count(self) -> int:
        return len(self._events)

    def start(self) -> None:
        self._recording = True
        self._start_time = time.time()
        self._events.clear()
        self._bus.subscribe(WILDCARD, self._on_event)

    def stop(self) -> None:
        self._recording = False
        self._bus.unsubscribe(WILDCARD, self._on_event)

    def _on_event(self, event: dict) -> None:
        if not self._recording:
            return
        self._events.append(event)

    def save(self) -> str:
        with open(self._path, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return self._path

    def get_events(self, event_type: str | None = None) -> list[dict]:
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.get("type") == event_type]

    def get_summary(self) -> dict:
        type_counts: dict[str, int] = {}
        for e in self._events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_events": len(self._events),
            "event_types": type_counts,
            "duration_sec": round(time.time() - self._start_time, 3) if self._start_time else 0,
        }

    @staticmethod
    def load(path: str) -> list[dict]:
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
