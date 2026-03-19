"""
PluginRegistry — centralized registry for discovering and creating
plugins and agents by name. Follows Whitepaper §4.1 plugin-first
architecture: modules register themselves, and the Engine instantiates
them by string key, avoiding import-time coupling.
"""

from __future__ import annotations

from typing import Any, Callable, Type


class PluginRegistry:
    """
    Maintains two namespaces — 'plugin' and 'agent' — each mapping a
    string name to a factory (usually a class).

    Usage::

        registry = PluginRegistry()
        registry.register_plugin("mock_camera", MockCamera)
        cam = registry.create_plugin("mock_camera", event_bus=bus, ...)

    Or use the decorator shorthand::

        @registry.plugin("mock_camera")
        class MockCamera(BasePlugin): ...
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Callable[..., Any]] = {}
        self._agents: dict[str, Callable[..., Any]] = {}

    @property
    def plugin_names(self) -> list[str]:
        return sorted(self._plugins)

    @property
    def agent_names(self) -> list[str]:
        return sorted(self._agents)

    def register_plugin(self, name: str, factory: Callable[..., Any]) -> None:
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory).__name__}")
        self._plugins[name] = factory

    def register_agent(self, name: str, factory: Callable[..., Any]) -> None:
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory).__name__}")
        self._agents[name] = factory

    def plugin(self, name: str) -> Callable:
        """Decorator to register a plugin class by name."""
        def wrapper(cls: Type) -> Type:
            self.register_plugin(name, cls)
            return cls
        return wrapper

    def agent(self, name: str) -> Callable:
        """Decorator to register an agent class by name."""
        def wrapper(cls: Type) -> Type:
            self.register_agent(name, cls)
            return cls
        return wrapper

    def create_plugin(self, name: str, **kwargs: Any) -> Any:
        if name not in self._plugins:
            raise KeyError(
                f"Unknown plugin '{name}'. "
                f"Available: {', '.join(self.plugin_names) or '(none)'}"
            )
        return self._plugins[name](**kwargs)

    def create_agent(self, name: str, **kwargs: Any) -> Any:
        if name not in self._agents:
            raise KeyError(
                f"Unknown agent '{name}'. "
                f"Available: {', '.join(self.agent_names) or '(none)'}"
            )
        return self._agents[name](**kwargs)

    def has_plugin(self, name: str) -> bool:
        return name in self._plugins

    def has_agent(self, name: str) -> bool:
        return name in self._agents

    def clear(self) -> None:
        self._plugins.clear()
        self._agents.clear()


default_registry = PluginRegistry()
