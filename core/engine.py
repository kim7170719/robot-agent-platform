"""
Engine — the main execution loop of IR-AAP.
Orchestrates: plugin.update() → agent.perceive() → agent.decide() → agent.act()
Runs in deterministic discrete steps (no threading).
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from plugins.base_plugin import BasePlugin
    from core.config import Config

from core.event_bus import EventBus, _make_event
from core.plugin_deps import DependencyError, resolve_order


class Engine:
    """
    Registers plugins and agents, then drives the
    collect → perceive → decide → act loop for N steps.

    Publishes ``engine.metrics`` events with per-step timing.

    Accepts an optional ``Config`` instance; when provided the
    engine reads ``engine.enable_metrics`` from it.
    """

    def __init__(
        self,
        event_bus: EventBus,
        enable_metrics: bool = True,
        config: Config | None = None,
    ) -> None:
        self._bus = event_bus
        self._config = config
        self._plugins: list[BasePlugin] = []
        self._agents: list[BaseAgent] = []
        self._running = False
        self._sim_paused = False
        self._step_count = 0
        self._step_times: list[float] = []
        self._step_lock = threading.RLock()

        if config is not None:
            self._enable_metrics = config.get("engine.enable_metrics", enable_metrics)
        else:
            self._enable_metrics = enable_metrics

    @property
    def config(self) -> Config | None:
        return self._config

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def step_times(self) -> list[float]:
        return list(self._step_times)

    @property
    def sim_paused(self) -> bool:
        """When True, the host loop should skip ``step()`` (API / WS pause)."""
        return self._sim_paused

    @sim_paused.setter
    def sim_paused(self, value: bool) -> None:
        self._sim_paused = bool(value)

    @property
    def step_lock(self) -> threading.RLock:
        """Re-entrant lock held during ``step()`` for thread-safe PyBullet access."""
        return self._step_lock

    def get_metrics(self) -> dict:
        if not self._step_times:
            return {"steps": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
        total = sum(self._step_times)
        return {
            "steps": len(self._step_times),
            "total_ms": round(total, 3),
            "avg_ms": round(total / len(self._step_times), 3),
            "max_ms": round(max(self._step_times), 3),
            "min_ms": round(min(self._step_times), 3),
        }

    def register_plugin(self, plugin: BasePlugin) -> None:
        self._plugins.append(plugin)

    def register_agent(self, agent: BaseAgent) -> None:
        self._agents.append(agent)

    def start(self) -> None:
        try:
            ordered = resolve_order(self._plugins)
            self._plugins = ordered
        except DependencyError as exc:
            warnings.warn(
                f"Engine: plugin dependency resolution failed ({exc!s}); "
                "using registration order.",
                RuntimeWarning,
                stacklevel=2,
            )
        for plugin in self._plugins:
            plugin.start()
        self._running = True

    def stop(self) -> None:
        self._running = False
        for plugin in self._plugins:
            plugin.stop()
        for agent in self._agents:
            agent.cleanup()

    def step(self) -> None:
        """Execute one full cycle of the loop, recording wall-clock time."""
        with self._step_lock:
            t0 = time.perf_counter()

            for plugin in self._plugins:
                try:
                    plugin.update()
                except Exception as exc:
                    warnings.warn(
                        f"Engine: plugin {plugin.plugin_id!r} raised {exc!r} during update; skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            for agent in self._agents:
                try:
                    agent.perceive()
                    agent.decide()
                    agent.act()
                except Exception as exc:
                    warnings.warn(
                        f"Engine: agent {agent.agent_id!r} raised {exc!r} during cycle; skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            self._step_count += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._step_times.append(elapsed_ms)

            if self._enable_metrics:
                self._bus.publish(_make_event("engine.metrics", "engine", {
                    "step": self._step_count,
                    "elapsed_ms": round(elapsed_ms, 3),
                }))

    def run(self, steps: int = 1, on_step: Callable[[int], None] | None = None) -> None:
        """Run the loop for *steps* iterations."""
        self.start()
        try:
            for _ in range(steps):
                if not self._running:
                    break
                self.step()
                if on_step is not None:
                    on_step(self._step_count)
        finally:
            self.stop()
