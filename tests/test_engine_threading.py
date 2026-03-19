"""Thread-safety tests for Engine.step_lock / concurrent step()."""

from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus
from core.engine import Engine


def test_concurrent_steps_match_total() -> None:
    bus = EventBus()
    engine = Engine(bus, enable_metrics=False)

    def worker(n: int) -> None:
        for _ in range(n):
            engine.step()

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert engine.step_count == 300
