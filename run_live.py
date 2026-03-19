"""
run_live.py — Launch the full IR-AAP platform with API + WebSocket + simulation.

Ports and bind address can be set via environment (see ``core.runtime_config``)
or a project-root ``.env`` file loaded at startup.

Stack composition: ``core.live_bootstrap.build_live_stack`` (YAML profile +
optional third-party ``iraap.register`` entry points).

Default services:
  - REST API:    http://localhost:8000  (Swagger: /docs)
  - WebSocket:   ws://localhost:8765
  - Dashboard:   http://localhost:8000/dashboard
"""

from __future__ import annotations

import asyncio
import json
import queue
import signal
import sys
import threading
import time

from core.live_bootstrap import build_live_stack
from core.recorder import Recorder
from core.runtime_config import get_api_host, get_api_port, get_ws_port, load_env_file
from core.event_schema import SchemaValidator
from core.state_machine import MissionFSM, Phase
from core.ws_commands import execute_ws_command

_shutdown = threading.Event()


def _request_shutdown(*_: object) -> None:
    _shutdown.set()


def _public_api_base(host: str, port: int) -> str:
    """Human-friendly URL when bound to all interfaces."""
    h = "localhost" if host in ("0.0.0.0", "::", "") else host
    return f"http://{h}:{port}"


def main(
    profile_path: str | None = None,
    *,
    world_backend: str | None = None,
) -> None:
    load_env_file()
    api_host = get_api_host()
    api_port = get_api_port()
    ws_port = get_ws_port()
    api_public = _public_api_base(api_host, api_port)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_shutdown)

    print("=" * 60)
    print("  IR-AAP Platform v2 — Full Stack Live Server")
    print("=" * 60)

    stack = build_live_stack(profile_path, world_backend_override=world_backend)
    bus = stack.bus
    cfg = stack.cfg
    world = stack.world
    engine = stack.engine
    metrics = stack.metrics
    gp = stack.goal_publisher
    ws_bridge = stack.ws_bridge
    hrm = stack.hrm
    omap = stack.occupancy_map

    ws_cmd_queue: queue.Queue = queue.Queue()

    schema_violations: list = []

    def _on_violation(v):
        schema_violations.append(v)

    sv = SchemaValidator(bus, strict=False, on_violation=_on_violation)

    fsm_scout = MissionFSM("scout", event_bus=bus)
    fsm_worker = MissionFSM("worker", event_bus=bus)
    fsm_scout.transition(Phase.NAVIGATE)
    fsm_worker.transition(Phase.NAVIGATE)

    recorder = Recorder(bus)

    print()
    print(f"  Profile:  {profile_path or '(default config/default_iraap.yaml)'}")
    print(f"  Backend:  {type(world).__name__}")
    print("  Extras:  HotReload, SchemaValidator, MissionFSM")
    print()

    def run_api():
        from api.server import create_app
        import uvicorn

        app = create_app(
            engine, bus, world,
            config=cfg,
            goal_publisher=gp,
            metrics_collector=metrics,
            hot_reload_manager=hrm,
            urdf_arm=stack.urdf_arm,
        )
        uvicorn.run(app, host=api_host, port=api_port, log_level="warning")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    print(f"  [OK] REST API:    {api_public}")
    print(f"        Swagger:    {api_public}/docs")

    def run_ws():
        clients: set = set()

        async def _serve():
            import websockets

            async def broadcast_loop():
                """Single drain → fan-out so multiple browser tabs all receive events."""
                while True:
                    events = ws_bridge.drain()
                    if events and clients:
                        dead = set()
                        for ev in events:
                            msg = json.dumps(ev, default=str)
                            for c in clients:
                                try:
                                    await c.send(msg)
                                except Exception:
                                    dead.add(c)
                        clients.difference_update(dead)
                    await asyncio.sleep(0.05)

            async def handler(ws):
                print("  [WS] Client connected")
                clients.add(ws)
                ws_bridge.connected_clients = len(clients)
                try:
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                            ws_cmd_queue.put(data)
                        except json.JSONDecodeError:
                            try:
                                await ws.send(json.dumps({
                                    "type": "ws.parse_error",
                                    "timestamp": int(time.time() * 1000),
                                    "source": "ws",
                                    "data": {"message": "invalid JSON"},
                                }, default=str))
                            except Exception:
                                pass
                except Exception:
                    pass
                finally:
                    clients.discard(ws)
                    ws_bridge.connected_clients = len(clients)
                    print("  [WS] Client disconnected")

            broadcaster = asyncio.create_task(broadcast_loop())
            try:
                server = await websockets.serve(handler, api_host, ws_port)
                await server.wait_closed()
            finally:
                broadcaster.cancel()
                try:
                    await broadcaster
                except asyncio.CancelledError:
                    pass

        asyncio.run(_serve())

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    ws_public = _public_api_base(api_host, ws_port).replace("http://", "ws://", 1)
    print(f"  [OK] WebSocket:   {ws_public}")
    print(f"  [OK] Dashboard:   {api_public}/dashboard")
    print()

    engine.start()
    recorder.start()
    print("  Simulation running (0.3s/step). Press Ctrl+C to stop.")
    print()

    step = 0
    try:
        while not _shutdown.is_set():
            while True:
                try:
                    cmd = ws_cmd_queue.get_nowait()
                except queue.Empty:
                    break
                execute_ws_command(
                    cmd,
                    engine=engine,
                    world=world,
                    bus=bus,
                    goal_publisher=gp,
                    config=cfg,
                )

            if not engine.sim_paused:
                engine.step()
                step += 1
                m = engine.get_metrics()
                occ = omap.cell_count()
                positions = {
                    rid: world.get_robot_position(rid)
                    for rid in world.get_robot_ids()
                }
                pos_str = "  ".join(
                    f"{rid}=({p[0]:.1f},{p[1]:.1f})" for rid, p in positions.items()
                )
                if step % 10 == 1 or step <= 5:
                    print(
                        f"  Step {step:4d} | {m['avg_ms']:.2f}ms avg "
                        f"| map: {occ['free']}F/{occ['occupied']}O "
                        f"| schema: {sv.violation_count} violations "
                        f"| {pos_str}"
                    )
                    sys.stdout.flush()
            time.sleep(0.3)
    except KeyboardInterrupt:
        _request_shutdown()
        print()
        print("  Stopping...")

    engine.stop()
    recorder.stop()
    recorder.save("live_session.jsonl")
    print(f"  Session saved: live_session.jsonl ({recorder.event_count} events)")
    print(f"  Metrics: {engine.get_metrics()}")
    print(f"  Schema checked: {sv.checked_count}, violations: {sv.violation_count}")
    print(f"  FSM scout: {fsm_scout.get_status()}")
    print(f"  FSM worker: {fsm_worker.get_status()}")
    world.disconnect()
    print("  Done.")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="IR-AAP full stack live server")
    _ap.add_argument("--profile", default=None, help="YAML profile path")
    _ap.add_argument("--mock-world", action="store_true", help="Use mock world (no PyBullet)")
    _args = _ap.parse_args()
    main(_args.profile, world_backend="mock" if _args.mock_world else None)
