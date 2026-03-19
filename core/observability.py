"""
Observability — structured JSON logging and metrics collection for IR-AAP.

Components:
  - StructuredLogger: wraps Python's logging module with JSON output and
    automatic context injection (source, timestamp, event_type).
  - MetricsCollector: lightweight counters, gauges, and histograms that
    can be exported to Prometheus text format or JSON.
  - ObservabilityPlugin: BasePlugin that subscribes to all events and
    feeds them into logger + metrics.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any

from core.event_bus import EventBus, WILDCARD
from plugins.base_plugin import BasePlugin


class StructuredLogger:
    """
    JSON-formatted logger backed by Python's logging module.
    Each log line is a JSON object with timestamp, level, source, and message.
    """

    def __init__(
        self,
        name: str = "iraap",
        level: int = logging.INFO,
        log_file: str | None = None,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler: logging.Handler
            if log_file:
                handler = logging.FileHandler(log_file, encoding="utf-8")
            else:
                handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)

        self._log_count = 0

    @property
    def log_count(self) -> int:
        return self._log_count

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def info(self, message: str, **extra: Any) -> None:
        self._logger.info(message, extra={"structured": extra})
        self._log_count += 1

    def warning(self, message: str, **extra: Any) -> None:
        self._logger.warning(message, extra={"structured": extra})
        self._log_count += 1

    def error(self, message: str, **extra: Any) -> None:
        self._logger.error(message, extra={"structured": extra})
        self._log_count += 1

    def debug(self, message: str, **extra: Any) -> None:
        self._logger.debug(message, extra={"structured": extra})
        self._log_count += 1

    def log_event(self, event: dict) -> None:
        self.info(
            f"event:{event.get('type', 'unknown')}",
            event_type=event.get("type"),
            source=event.get("source"),
            timestamp=event.get("timestamp"),
        )


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if structured:
            obj.update(structured)
        return json.dumps(obj, default=str)


class MetricsCollector:
    """
    Lightweight metrics store with counters, gauges, and histograms.
    Exports to Prometheus text exposition format or JSON.
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._histogram_cap = 1000

    def inc(self, name: str, value: float = 1.0) -> None:
        self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        h = self._histograms[name]
        h.append(value)
        if len(h) > self._histogram_cap:
            h[:] = h[-self._histogram_cap:]

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float | None:
        return self._gauges.get(name)

    def get_histogram_summary(self, name: str) -> dict:
        h = self._histograms.get(name, [])
        if not h:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
        return {
            "count": len(h),
            "sum": round(sum(h), 4),
            "avg": round(sum(h) / len(h), 4),
            "min": round(min(h), 4),
            "max": round(max(h), 4),
        }

    def to_json(self) -> dict:
        result: dict[str, Any] = {}
        for name, val in sorted(self._counters.items()):
            result[f"counter_{name}"] = val
        for name, val in sorted(self._gauges.items()):
            result[f"gauge_{name}"] = val
        for name in sorted(self._histograms):
            result[f"histogram_{name}"] = self.get_histogram_summary(name)
        return result

    def to_prometheus(self) -> str:
        lines: list[str] = []
        for name, val in sorted(self._counters.items()):
            safe = name.replace(".", "_")
            lines.append(f"# TYPE {safe} counter")
            lines.append(f"{safe} {val}")
        for name, val in sorted(self._gauges.items()):
            safe = name.replace(".", "_")
            lines.append(f"# TYPE {safe} gauge")
            lines.append(f"{safe} {val}")
        for name in sorted(self._histograms):
            safe = name.replace(".", "_")
            summary = self.get_histogram_summary(name)
            lines.append(f"# TYPE {safe} summary")
            lines.append(f"{safe}_count {summary['count']}")
            lines.append(f"{safe}_sum {summary['sum']}")
        return "\n".join(lines) + "\n" if lines else ""

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class ObservabilityPlugin(BasePlugin):
    """
    Plugin that subscribes to all EventBus events and feeds them
    into a StructuredLogger and MetricsCollector.
    """

    def __init__(
        self,
        event_bus: EventBus,
        logger: StructuredLogger | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        super().__init__(event_bus, plugin_id="observability")
        self._logger = logger or StructuredLogger(name="iraap-obs", level=logging.DEBUG)
        self._metrics = metrics or MetricsCollector()

    @property
    def structured_logger(self) -> StructuredLogger:
        return self._logger

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    def start(self) -> None:
        super().start()
        self._bus.subscribe(WILDCARD, self._on_event)

    def stop(self) -> None:
        super().stop()
        self._bus.unsubscribe(WILDCARD, self._on_event)

    def _on_event(self, event: dict) -> None:
        event_type = event.get("type", "unknown")
        self._metrics.inc(f"events.{event_type}")
        self._metrics.inc("events.total")

        if event_type == "engine.metrics":
            elapsed = event.get("data", {}).get("elapsed_ms", 0)
            self._metrics.observe("step_time_ms", elapsed)
            self._metrics.set_gauge("last_step_ms", elapsed)
            step = event.get("data", {}).get("step", 0)
            self._metrics.set_gauge("step_count", step)

    def on_update(self) -> None:
        pass

    def get_status(self) -> dict:
        return {
            "log_count": self._logger.log_count,
            "metrics": self._metrics.to_json(),
        }
