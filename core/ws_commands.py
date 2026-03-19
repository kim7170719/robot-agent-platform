"""
WebSocket command executor — processes JSON commands from dashboard / clients.

Commands are queued on a background WebSocket thread and executed on the
simulation thread (see ``run_live.py``). Each command publishes a
``ws.command.ack`` event with ``ok`` and result details.

Protocol (JSON text frame)::

    {"cmd": "ping"}
    {"cmd": "step", "steps": 5}
    {"cmd": "pause"} / {"cmd": "resume"}
    {"cmd": "reset"}
    {"cmd": "goal", "x": 1.0, "y": 2.0, "z": 0.31}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.event_bus import _make_event

if TYPE_CHECKING:
    from core.engine import Engine
    from core.config import Config
    from simulator.physics_world import PhysicsWorld


def execute_ws_command(
    cmd: dict[str, Any],
    *,
    engine: Engine,
    world: PhysicsWorld,
    bus,
    goal_publisher=None,
    config: Config | None = None,
) -> None:
    """Execute one command dict and publish ``ws.command.ack`` on *bus*."""
    c = (cmd or {}).get("cmd", "")
    ok = True
    detail: dict[str, Any] = {"cmd": c}

    try:
        if c == "ping":
            detail["pong"] = True

        elif c == "pause":
            engine.sim_paused = True

        elif c == "resume":
            engine.sim_paused = False

        elif c == "step":
            n = int(cmd.get("steps", 1))
            n = max(1, min(1000, n))
            with engine.step_lock:
                if not engine._running:
                    engine.start()
                for _ in range(n):
                    engine.step()
                detail["steps_executed"] = n
                detail["total_steps"] = engine.step_count

        elif c == "reset":
            with engine.step_lock:
                world.reset()

        elif c == "goal":
            if goal_publisher is None:
                ok = False
                detail["reason"] = "no goal publisher"
            else:
                x = float(cmd.get("x", 0))
                y = float(cmd.get("y", 0))
                z = float(cmd.get("z", 0.31))
                goal_publisher.set_target(x, y, z)
                detail["goal"] = [x, y, z]

        elif c == "config_patch" and config is not None:
            key = cmd.get("key")
            if not key:
                ok = False
                detail["reason"] = "missing key"
            else:
                config.set(str(key), cmd.get("value"))
                detail["key"] = key

        else:
            ok = False
            detail["reason"] = f"unknown or unsupported cmd: {c!r}"

    except Exception as exc:  # noqa: BLE001 — surface to client via ack
        ok = False
        detail["reason"] = str(exc)

    detail["ok"] = ok
    bus.publish(_make_event("ws.command.ack", "ws_commands", detail))
