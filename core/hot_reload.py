"""
HotReloadManager — dynamic plugin load/unload at runtime.

Enables the Engine to add or remove plugins and agents while the
simulation is running, without restarting. Follows Whitepaper §4.1
plugin-first architecture.

Events published:
  - ``system.plugin_loaded``   when a plugin is hot-loaded
  - ``system.plugin_unloaded`` when a plugin is hot-unloaded
  - ``system.agent_loaded``    when an agent is hot-loaded
  - ``system.agent_unloaded``  when an agent is hot-unloaded
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.base_plugin import BasePlugin
    from agents.base_agent import BaseAgent

from core.event_bus import EventBus, _make_event


class HotReloadManager:
    """
    Wraps an Engine and exposes ``load_plugin``, ``unload_plugin``,
    ``load_agent``, ``unload_agent`` for runtime changes.
    """

    def __init__(self, engine, event_bus: EventBus) -> None:
        self._engine = engine
        self._bus = event_bus
        self._load_count = 0
        self._unload_count = 0

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def unload_count(self) -> int:
        return self._unload_count

    def load_plugin(self, plugin: BasePlugin) -> None:
        """Add and start a plugin while the engine is running."""
        self._engine._plugins.append(plugin)
        if self._engine._running:
            plugin.start()
        self._load_count += 1

        self._bus.publish(_make_event("system.plugin_loaded", "hot_reload", {
            "plugin_id": plugin.plugin_id,
            "total_plugins": len(self._engine._plugins),
        }))

    def unload_plugin(self, plugin_id: str) -> bool:
        """Stop and remove a plugin by ID. Returns True if found."""
        for i, p in enumerate(self._engine._plugins):
            if p.plugin_id == plugin_id:
                p.stop()
                self._engine._plugins.pop(i)
                self._unload_count += 1

                self._bus.publish(_make_event("system.plugin_unloaded", "hot_reload", {
                    "plugin_id": plugin_id,
                    "total_plugins": len(self._engine._plugins),
                }))
                return True
        return False

    def load_agent(self, agent: BaseAgent) -> None:
        """Add an agent while the engine is running."""
        self._engine._agents.append(agent)
        self._load_count += 1

        self._bus.publish(_make_event("system.agent_loaded", "hot_reload", {
            "agent_id": agent.agent_id,
            "total_agents": len(self._engine._agents),
        }))

    def unload_agent(self, agent_id: str) -> bool:
        """Cleanup and remove an agent by ID. Returns True if found."""
        for i, a in enumerate(self._engine._agents):
            if a.agent_id == agent_id:
                a.cleanup()
                self._engine._agents.pop(i)
                self._unload_count += 1

                self._bus.publish(_make_event("system.agent_unloaded", "hot_reload", {
                    "agent_id": agent_id,
                    "total_agents": len(self._engine._agents),
                }))
                return True
        return False

    def swap_plugin(self, old_id: str, new_plugin: BasePlugin) -> bool:
        """Atomically replace a plugin: unload old, load new."""
        removed = self.unload_plugin(old_id)
        self.load_plugin(new_plugin)
        return removed

    def get_status(self) -> dict:
        return {
            "plugins": [p.plugin_id for p in self._engine._plugins],
            "agents": [a.agent_id for a in self._engine._agents],
            "total_loads": self._load_count,
            "total_unloads": self._unload_count,
        }
