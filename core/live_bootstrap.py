"""
Assemble a runnable live stack (world, engine, plugins, agents) from YAML profile.

Used by ``run_live.py`` and API integration tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from core.builtin_entrypoints import register_builtin
from core.config import Config
from core.engine import Engine
from core.event_bus import EventBus
from core.ext_discovery import load_entrypoint_registers
from core.hot_reload import HotReloadManager
from core.observability import MetricsCollector, ObservabilityPlugin
from core.profile_loader import (
    build_world_from_profile,
    default_profile_path,
    instantiate_plugins_and_agents,
    load_profile,
)
from core.registry import PluginRegistry
from core.ws_bridge import WebSocketBridge


@dataclass
class LiveStack:
    bus: EventBus
    cfg: Config
    world: Any
    engine: Engine
    metrics: MetricsCollector
    goal_publisher: Any
    ws_bridge: WebSocketBridge
    hrm: HotReloadManager
    occupancy_map: Any
    urdf_arm: Any | None = None


def build_live_stack(
    profile_path: str | None = None,
    *,
    world_backend_override: str | None = None,
    bus_max_history: int = 5000,
) -> LiveStack:
    """
    Load profile (``IRAAP_PROFILE`` env, *profile_path*, or ``config/default_iraap.yaml``),
    discover entry points, build world + engine + plugins + agents.

    If *world_backend_override* is the string ``mock`` and no profile path or env is set,
    uses ``config/mock_api_iraap.yaml`` (PyBullet-free plugins for tests / quick runs).
    """
    env_prof = os.environ.get("IRAAP_PROFILE")
    if world_backend_override == "mock" and profile_path is None and not env_prof:
        path = os.path.join(os.path.dirname(default_profile_path()), "mock_api_iraap.yaml")
    else:
        path = profile_path or env_prof or default_profile_path()
    profile = load_profile(path)
    if not profile:
        profile = load_profile(default_profile_path())

    world_spec = dict(profile.get("world") or {})
    if world_backend_override:
        world_spec["backend"] = world_backend_override

    registry = PluginRegistry()
    register_builtin(registry)
    load_entrypoint_registers(registry)

    bus = EventBus(max_history=bus_max_history)
    world = build_world_from_profile(world_spec)

    urdf_arm = None
    man = profile.get("manipulator") or {}
    if man.get("enabled"):
        cid = getattr(world, "client_id", None)
        if cid is not None and int(cid) >= 0:
            from simulator.urdf_robot import URDFRobot

            pos = (
                float(man.get("x", 2.0)),
                float(man.get("y", 0.0)),
                float(man.get("z", 0.15)),
            )
            upath = man.get("urdf_path")
            if upath is None or (isinstance(upath, str) and not str(upath).strip()):
                upath = None
            elif isinstance(upath, str):
                upath = upath.strip()
            urdf_arm = URDFRobot(
                int(cid),
                urdf_path=upath,
                position=pos,
                robot_id=str(man.get("id", "box_arm")),
            )

    eng_cfg = profile.get("engine") or {}
    cfg_dict = eng_cfg.get("config") or {"engine": {"enable_metrics": True}}
    cfg = Config(cfg_dict)

    engine = Engine(bus, config=cfg)
    metrics = MetricsCollector()

    goal_pub, omap = instantiate_plugins_and_agents(
        profile, registry=registry, bus=bus, world=world, engine=engine
    )

    obs_plugin = ObservabilityPlugin(bus, metrics=metrics)
    ws_bridge = WebSocketBridge(bus)
    engine.register_plugin(obs_plugin)
    engine.register_plugin(ws_bridge)

    hrm = HotReloadManager(engine, bus)

    g = profile.get("goal") or {}
    if goal_pub is not None:
        goal_pub.set_target(float(g.get("x", 0)), float(g.get("y", 0)), float(g.get("z", 0.31)))

    if omap is None:
        raise RuntimeError("Profile must include occupancy_map plugin")

    return LiveStack(
        bus=bus,
        cfg=cfg,
        world=world,
        engine=engine,
        metrics=metrics,
        goal_publisher=goal_pub,
        ws_bridge=ws_bridge,
        hrm=hrm,
        occupancy_map=omap,
        urdf_arm=urdf_arm,
    )
