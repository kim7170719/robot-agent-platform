"""core — Event Bus, Engine, Config, Recorder, Replayer, Registry, WebSocketBridge,
HotReload, Observability, Schema, Deps, FSM."""

from core.event_bus import EventBus, _make_event, WILDCARD
from core.engine import Engine
from core.config import Config
from core.recorder import Recorder
from core.replayer import Replayer
from core.registry import PluginRegistry, default_registry
from core.ws_bridge import WebSocketBridge
from core.hot_reload import HotReloadManager
from core.observability import StructuredLogger, MetricsCollector, ObservabilityPlugin
from core.event_schema import SCHEMAS, SchemaValidator, validate_event_data, ValidationError
from core.plugin_deps import resolve_order, build_graph, check_health, DependencyError
from core.state_machine import MissionFSM, Phase, Transition
from core.ws_commands import execute_ws_command

__all__ = [
    "EventBus",
    "_make_event",
    "WILDCARD",
    "Engine",
    "Config",
    "Recorder",
    "Replayer",
    "PluginRegistry",
    "default_registry",
    "WebSocketBridge",
    "HotReloadManager",
    "StructuredLogger",
    "MetricsCollector",
    "ObservabilityPlugin",
    "SCHEMAS",
    "SchemaValidator",
    "validate_event_data",
    "ValidationError",
    "resolve_order",
    "build_graph",
    "check_health",
    "DependencyError",
    "MissionFSM",
    "Phase",
    "Transition",
    "execute_ws_command",
]
