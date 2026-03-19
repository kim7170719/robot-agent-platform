"""FastAPI integration tests (v1 + mock world, no PyBullet)."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.server import create_app
from core.live_bootstrap import build_live_stack


def _mock_profile_path() -> str:
    root = os.path.join(os.path.dirname(__file__), "..", "config", "mock_api_iraap.yaml")
    return os.path.normpath(root)


@pytest.fixture
def mock_client() -> TestClient:
    stack = build_live_stack(_mock_profile_path())
    app = create_app(
        stack.engine,
        stack.bus,
        stack.world,
        config=stack.cfg,
        goal_publisher=stack.goal_publisher,
        metrics_collector=stack.metrics,
        hot_reload_manager=stack.hrm,
    )
    with TestClient(app) as client:
        yield client


def test_v1_health(mock_client: TestClient) -> None:
    r = mock_client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "step_count" in body


def test_legacy_health_alias(mock_client: TestClient) -> None:
    r = mock_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_v1_step_and_state(mock_client: TestClient) -> None:
    r1 = mock_client.post("/api/v1/step", json={"steps": 2})
    assert r1.status_code == 200
    assert r1.json()["total_steps"] == 2
    r2 = mock_client.get("/api/v1/state")
    assert r2.status_code == 200
    body = r2.json()
    assert body["step_count"] == 2
    assert "dynamics" in body and "dynamic_count" in body


def test_dynamics_box_api(mock_client: TestClient) -> None:
    r = mock_client.post(
        "/api/v1/dynamics/box",
        json={"x": 0.5, "y": 0, "z": 1.0, "mass": 0.2, "half_extent": 0.06},
    )
    assert r.status_code == 200
    assert r.json().get("dynamic_count", 0) >= 1
    st = mock_client.get("/api/v1/state").json()
    assert st["dynamic_count"] >= 1
    assert mock_client.delete("/api/v1/dynamics").status_code == 200
    assert mock_client.get("/api/v1/state").json()["dynamic_count"] == 0


def test_http_error_shape(mock_client: TestClient) -> None:
    r = mock_client.post("/api/v1/step", json={"steps": 0})
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert "code" in data["error"]
    assert "message" in data["error"]


def test_auto_mock_profile_when_override() -> None:
    stack = build_live_stack(None, world_backend_override="mock")
    assert stack.world.__class__.__name__ == "MockPhysicsWorld"


def test_openapi_lists_v1(mock_client: TestClient) -> None:
    r = mock_client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert "/api/v1/health" in paths
    assert "/api/health" not in paths
