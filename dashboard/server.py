"""
Dashboard WebSocket Server — standalone WebSocket broadcast for EventBus events.

**Preferred full-stack entry point:** run ``python run_live.py`` and open
``http://localhost:8000/dashboard`` (FastAPI serves HTML; WS on port 8765).
This module is optional for development without the REST API.

For bidirectional commands (pause/step/goal) use the WebSocket protocol
documented in ``core/ws_commands.py`` when using ``run_live.py``.

Usage:
    python -m dashboard.server            # WS-only standalone
    from dashboard.server import start_dashboard
"""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    import websockets
    import websockets.server
except ImportError:
    websockets = None  # type: ignore[assignment]

from core.ws_bridge import WebSocketBridge

_DASHBOARD_DIR = Path(__file__).parent
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_WS_PORT = 8765
_DEFAULT_HTTP_PORT = 8080
_BROADCAST_INTERVAL = 0.05


async def _broadcast_loop(bridge: WebSocketBridge, clients: set) -> None:
    """Periodically drain the bridge queue and send to all clients."""
    while True:
        events = bridge.drain()
        if events and clients:
            messages = [bridge.serialize_event(ev) for ev in events]
            payload = "[" + ",".join(messages) + "]"
            dead = set()
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            clients -= dead
            bridge.connected_clients = len(clients)
        await asyncio.sleep(_BROADCAST_INTERVAL)


async def _ws_handler(
    websocket, bridge: WebSocketBridge, clients: set,
) -> None:
    clients.add(websocket)
    bridge.connected_clients = len(clients)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.discard(websocket)
        bridge.connected_clients = len(clients)


async def run_dashboard_server(
    bridge: WebSocketBridge,
    host: str = _DEFAULT_HOST,
    ws_port: int = _DEFAULT_WS_PORT,
) -> None:
    """Start the WebSocket server (blocks until cancelled)."""
    if websockets is None:
        raise ImportError("websockets package is required: pip install websockets")

    clients: set = set()
    broadcast_task = asyncio.create_task(_broadcast_loop(bridge, clients))

    async with websockets.server.serve(
        lambda ws: _ws_handler(ws, bridge, clients),
        host, ws_port,
    ):
        print(f"[Dashboard] WebSocket server on ws://{host}:{ws_port}")
        await asyncio.Future()

    broadcast_task.cancel()


def start_dashboard(
    bridge: WebSocketBridge,
    host: str = _DEFAULT_HOST,
    ws_port: int = _DEFAULT_WS_PORT,
) -> None:
    """Convenience blocking entry point."""
    asyncio.run(run_dashboard_server(bridge, host, ws_port))
