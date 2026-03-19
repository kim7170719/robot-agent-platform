"""
REST API Server — FastAPI application for remote control and observation
of the IR-AAP simulation.

Routes are mounted at **/api/v1/...** (preferred) and **/api/...** (legacy alias).

Usage:
    from api.server import create_app, start_api
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.sim_routes import build_sim_router

if TYPE_CHECKING:
    from core.config import Config
    from core.engine import Engine
    from core.event_bus import EventBus
    from core.observability import MetricsCollector


class _AppState:
    engine: Engine | None = None
    event_bus: EventBus | None = None
    world: Any = None
    config: Config | None = None
    goal_publisher = None
    metrics_collector: MetricsCollector | None = None
    hot_reload_manager = None
    urdf_arm: Any = None


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app(
    engine: Engine,
    event_bus: EventBus,
    world: Any,
    config: Config | None = None,
    goal_publisher=None,
    metrics_collector: MetricsCollector | None = None,
    hot_reload_manager=None,
    urdf_arm: Any | None = None,
) -> FastAPI:

    app = FastAPI(title="IR-AAP API", version="2.0.0")

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                },
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request body or query",
                    "details": exc.errors(),
                },
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = _AppState()
    state.engine = engine
    state.event_bus = event_bus
    state.world = world
    state.config = config
    state.goal_publisher = goal_publisher
    state.metrics_collector = metrics_collector
    state.hot_reload_manager = hot_reload_manager
    state.urdf_arm = urdf_arm
    app.state.sim = state  # type: ignore[attr-defined]

    def _s() -> _AppState:
        return app.state.sim  # type: ignore[attr-defined]

    project_root_dir = _project_root()

    @app.get("/")
    def root():
        return {
            "name": "IR-AAP API",
            "version": "2.0.0",
            "docs": "/docs",
            "dashboard": "/dashboard",
            "wizard": "/wizard",
            "api_v1": "/api/v1",
            "api_legacy": "/api",
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        html_path = os.path.join(project_root_dir, "dashboard", "index.html")
        if not os.path.exists(html_path):
            raise HTTPException(404, "dashboard/index.html not found")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @app.get("/wizard", response_class=HTMLResponse)
    def mission_wizard_page():
        html_path = os.path.join(project_root_dir, "dashboard", "mission_wizard.html")
        if not os.path.exists(html_path):
            raise HTTPException(404, "dashboard/mission_wizard.html not found")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    v1 = build_sim_router(_s, project_root=project_root_dir)
    legacy = build_sim_router(_s, project_root=project_root_dir)
    app.include_router(v1, prefix="/api/v1", tags=["v1"])
    app.include_router(legacy, prefix="/api", include_in_schema=False)

    return app


def start_api(
    engine: Engine,
    event_bus: EventBus,
    world: Any,
    config: Config | None = None,
    goal_publisher=None,
    metrics_collector: MetricsCollector | None = None,
    hot_reload_manager=None,
    urdf_arm: Any | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Blocking convenience entry point."""
    import uvicorn

    app = create_app(
        engine, event_bus, world, config, goal_publisher,
        metrics_collector, hot_reload_manager, urdf_arm=urdf_arm,
    )
    uvicorn.run(app, host=host, port=port)
