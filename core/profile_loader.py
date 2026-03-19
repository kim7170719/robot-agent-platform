"""Load YAML stack profiles for live_bootstrap."""

from __future__ import annotations

import os
from typing import Any

import yaml

from core.registry import PluginRegistry


def default_profile_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "config", "default_iraap.yaml")


def load_profile(path: str | None) -> dict[str, Any]:
    """Load a profile YAML file; returns empty dict if *path* is None or missing."""
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_world_from_profile(world_spec: dict[str, Any]) -> Any:
    """Instantiate ``PhysicsWorld`` or ``MockPhysicsWorld`` from profile ``world`` section."""
    backend = (world_spec.get("backend") or "pybullet").strip().lower()
    gui = bool(world_spec.get("gui", False))

    phy = world_spec.get("physics") or {}
    timestep = float(phy.get("timestep", 1.0 / 240.0))
    gravity_z = float(phy.get("gravity_z", -9.81))

    if backend == "mock":
        from simulator.mock_physics_world import MockPhysicsWorld

        world = MockPhysicsWorld()
    else:
        from simulator.physics_world import PhysicsWorld

        world = PhysicsWorld(gui=gui, timestep=timestep, gravity=gravity_z)

    for bot in world_spec.get("robots", []) or []:
        rid = bot["id"]
        x = float(bot.get("x", 0))
        y = float(bot.get("y", 0))
        z = bot.get("z")
        if rid == "default":
            world.set_robot_position(x, y, float(z) if z is not None else None, "default")
        else:
            world.add_robot(rid, x=x, y=y, z=float(z) if z is not None else None)

    for obs in world_spec.get("obstacles", []) or []:
        world.place_obstacle(float(obs["x"]), float(obs["y"]), float(obs.get("z", 0.5)))

    for dyn in world_spec.get("dynamics", []) or []:
        if hasattr(world, "spawn_dynamic_box"):
            world.spawn_dynamic_box(
                float(dyn["x"]),
                float(dyn["y"]),
                float(dyn["z"]),
                half_extent=float(dyn.get("half_extent", 0.1)),
                mass=float(dyn.get("mass", 1.0)),
            )

    return world


def instantiate_plugins_and_agents(
    profile: dict[str, Any],
    *,
    registry: PluginRegistry,
    bus: Any,
    world: Any,
    engine: Any,
) -> tuple[Any | None, Any]:
    """
    Create plugins/agents from profile, register on *engine*.
    Returns ``(goal_publisher, occupancy_map_plugin)`` when present.
    """
    goal_pub = None
    omap = None

    for spec in profile.get("plugins", []) or []:
        ptype = spec["type"]
        kwargs = dict(spec.get("kwargs") or {})
        if ptype == "goal_publisher":
            inst = registry.create_plugin(ptype, event_bus=bus, **kwargs)
            goal_pub = inst
        elif ptype == "occupancy_map":
            inst = registry.create_plugin(ptype, event_bus=bus, **kwargs)
            omap = inst
        else:
            inst = registry.create_plugin(ptype, event_bus=bus, world=world, **kwargs)
        engine.register_plugin(inst)

    for spec in profile.get("agents", []) or []:
        atype = spec["type"]
        kwargs = dict(spec.get("kwargs") or {})
        agent = registry.create_agent(atype, event_bus=bus, world=world, **kwargs)
        engine.register_agent(agent)

    return goal_pub, omap
