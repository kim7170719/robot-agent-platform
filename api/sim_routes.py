"""Simulation REST routes (mounted at ``/api/v1`` and legacy ``/api``)."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

import pybullet as p
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from api.schemas import (
    ConfigPatchRequest,
    DynamicBoxRequest,
    GoalRequest,
    GripperRequest,
    MoveRobotRequest,
    ObstacleRequest,
    RobotRequest,
    RobotTeleportRequest,
    StepRequest,
)


def _safe_session_basename(name: str) -> str:
    if not name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "invalid session file name")
    base = os.path.basename(name)
    if not base.endswith(".jsonl"):
        raise HTTPException(400, "file must end with .jsonl")
    return base


def build_sim_router(_s: Callable[[], Any], *, project_root: str) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health():
        s = _s()
        with s.engine.step_lock:
            return {
                "status": "ok",
                "paused": s.engine.sim_paused,
                "step_count": s.engine.step_count,
                "robot_count": s.world.robot_count,
            }

    @router.get("/state")
    def get_state():
        s = _s()
        with s.engine.step_lock:
            robots = {}
            for rid in s.world.get_robot_ids():
                pos = s.world.get_robot_position(rid)
                robots[rid] = {"position": list(pos)}
            dyn = (
                s.world.get_dynamic_objects()
                if hasattr(s.world, "get_dynamic_objects")
                else []
            )
            return {
                "robots": robots,
                "robot_count": s.world.robot_count,
                "obstacle_count": len(s.world._obstacle_ids),
                "obstacles": s.world.get_obstacles(),
                "dynamics": dyn,
                "dynamic_count": len(dyn),
                "step_count": s.engine.step_count,
                "paused": s.engine.sim_paused,
                "plugins": [p.plugin_id for p in s.engine._plugins],
                "agents": [a.agent_id for a in s.engine._agents],
            }

    @router.post("/step")
    def post_step(req: StepRequest):
        s = _s()
        if req.steps < 1 or req.steps > 1000:
            raise HTTPException(400, "steps must be between 1 and 1000")
        with s.engine.step_lock:
            before = s.engine.step_count
            if not s.engine._running:
                s.engine.start()
            for _ in range(req.steps):
                s.engine.step()
            total = s.engine.step_count
        return {
            "steps_executed": total - before,
            "total_steps": total,
        }

    @router.post("/reset")
    def post_reset():
        s = _s()
        with s.engine.step_lock:
            s.world.reset()
            rids = s.world.get_robot_ids()
        return {"status": "reset", "robots": rids}

    @router.post("/pause")
    def post_pause():
        s = _s()
        with s.engine.step_lock:
            s.engine.sim_paused = True
        return {"paused": True}

    @router.post("/resume")
    def post_resume():
        s = _s()
        with s.engine.step_lock:
            s.engine.sim_paused = False
        return {"paused": False}

    @router.post("/goal")
    def post_goal(req: GoalRequest):
        s = _s()
        if s.goal_publisher is None:
            raise HTTPException(400, "No goal publisher configured")
        s.goal_publisher.set_target(req.x, req.y, req.z)
        return {"goal": [req.x, req.y, req.z]}

    @router.post("/robots/add")
    def add_robot(req: RobotRequest):
        s = _s()
        with s.engine.step_lock:
            try:
                s.world.add_robot(req.robot_id, x=req.x, y=req.y, z=req.z)
            except ValueError as e:
                raise HTTPException(400, str(e))
            rc = s.world.robot_count
        return {
            "added": req.robot_id,
            "position": [req.x, req.y, req.z],
            "robot_count": rc,
        }

    @router.delete("/robots/{robot_id}")
    def remove_robot(robot_id: str):
        s = _s()
        with s.engine.step_lock:
            try:
                s.world.remove_robot(robot_id)
            except ValueError as e:
                raise HTTPException(400, str(e))
            rc = s.world.robot_count
        return {"removed": robot_id, "robot_count": rc}

    @router.post("/robots/move")
    def move_robot(req: MoveRobotRequest):
        s = _s()
        with s.engine.step_lock:
            try:
                success = s.world.move_robot(req.direction, req.robot_id)
            except (ValueError, KeyError) as e:
                raise HTTPException(400, str(e))
            pos = s.world.get_robot_position(req.robot_id)
        return {"success": success, "position": list(pos), "direction": req.direction}

    @router.post("/robots/position")
    def set_robot_position_api(req: RobotTeleportRequest):
        s = _s()
        if not hasattr(s.world, "set_robot_position"):
            raise HTTPException(400, "World does not support set_robot_position")
        with s.engine.step_lock:
            try:
                s.world.set_robot_position(req.x, req.y, req.z, req.robot_id)
            except TypeError as e:
                raise HTTPException(
                    400,
                    "Robot teleport needs a 3D physics world (PyBullet / mock API world).",
                ) from e
            except (ValueError, KeyError) as e:
                raise HTTPException(400, str(e))
            rid = req.robot_id if req.robot_id is not None else "default"
            pos = s.world.get_robot_position(rid)
        return {"robot_id": rid, "position": list(pos)}

    @router.post("/obstacles/add")
    def add_obstacle(req: ObstacleRequest):
        s = _s()
        with s.engine.step_lock:
            body_id = s.world.place_obstacle(req.x, req.y, req.z)
            oc = len(s.world._obstacle_ids)
        return {
            "obstacle_id": body_id,
            "position": [req.x, req.y, req.z],
            "obstacle_count": oc,
        }

    @router.delete("/obstacles/clear")
    def clear_obstacles():
        s = _s()
        with s.engine.step_lock:
            s.world.clear_obstacles()
        return {"obstacle_count": 0}

    @router.post("/dynamics/box")
    def spawn_dynamic_box_api(req: DynamicBoxRequest):
        s = _s()
        if not hasattr(s.world, "spawn_dynamic_box"):
            raise HTTPException(400, "World does not support dynamic bodies")
        if req.mass <= 0 or req.half_extent <= 0:
            raise HTTPException(400, "mass and half_extent must be positive")
        with s.engine.step_lock:
            bid = s.world.spawn_dynamic_box(
                req.x,
                req.y,
                req.z,
                mass=req.mass,
                half_extent=req.half_extent,
            )
            dyn = (
                s.world.get_dynamic_objects()
                if hasattr(s.world, "get_dynamic_objects")
                else []
            )
        return {"body_id": bid, "dynamic_count": len(dyn)}

    @router.delete("/dynamics")
    def clear_dynamics():
        s = _s()
        if not hasattr(s.world, "clear_dynamic_objects"):
            raise HTTPException(400, "World does not support dynamic bodies")
        with s.engine.step_lock:
            s.world.clear_dynamic_objects()
        return {"dynamic_count": 0, "cleared": True}

    @router.get("/manipulator/state")
    def manipulator_state():
        s = _s()
        arm = getattr(s, "urdf_arm", None)
        if arm is None:
            raise HTTPException(404, "No manipulator configured")
        with s.engine.step_lock:
            return arm.get_state()

    @router.post("/manipulator/gripper")
    def manipulator_gripper(req: GripperRequest):
        s = _s()
        arm = getattr(s, "urdf_arm", None)
        if arm is None:
            raise HTTPException(404, "No manipulator configured")
        cid = getattr(s.world, "client_id", -1)
        with s.engine.step_lock:
            arm.set_gripper_openness(req.openness)
            if cid is not None and int(cid) >= 0:
                for _ in range(120):
                    p.stepSimulation(physicsClientId=int(cid))
            return {"openness": float(req.openness), "state": arm.get_state()}

    @router.get("/config")
    def get_config():
        s = _s()
        if s.config is None:
            return {"config": {}}
        return {"config": s.config.to_dict()}

    @router.patch("/config")
    def patch_config(req: ConfigPatchRequest):
        s = _s()
        if s.config is None:
            raise HTTPException(400, "No config object available")
        with s.engine.step_lock:
            s.config.set(req.key, req.value)
            cfg = s.config.to_dict()
        return {"key": req.key, "value": req.value, "config": cfg}

    @router.post("/config/save")
    def save_config_file(filename: str = "iraap_config.yaml"):
        s = _s()
        if s.config is None:
            raise HTTPException(400, "No config object available")
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "invalid filename")
        base = os.path.basename(filename)
        if not base:
            raise HTTPException(400, "empty filename")
        path = os.path.join(project_root, base)
        s.config.save(path)
        return {"saved": base, "path": path}

    @router.get("/replay/sessions")
    def replay_sessions():
        try:
            names = sorted(f for f in os.listdir(project_root) if f.endswith(".jsonl"))
        except OSError:
            names = []
        return {"sessions": names, "directory": project_root}

    @router.get("/replay/events")
    def replay_events(file: str, offset: int = 0, limit: int = 200):
        base = _safe_session_basename(file)
        limit = min(max(1, limit), 5000)
        offset = max(0, offset)
        path = os.path.join(project_root, base)
        if not os.path.isfile(path):
            raise HTTPException(404, f"not found: {base}")
        events: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if len(events) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {
            "file": base,
            "offset": offset,
            "returned": len(events),
            "events": events,
        }

    @router.get("/metrics")
    def get_metrics():
        s = _s()
        with s.engine.step_lock:
            return s.engine.get_metrics()

    @router.get("/events")
    def get_events(event_type: str | None = None, limit: int = 50):
        s = _s()
        limit = min(max(1, limit), 500)
        history = s.event_bus.get_history(event_type)
        return {"events": history[-limit:], "total": len(history)}

    @router.get("/metrics/prometheus", response_class=PlainTextResponse)
    def get_prometheus_metrics():
        s = _s()
        if s.metrics_collector is None:
            return ""
        return s.metrics_collector.to_prometheus()

    @router.get("/metrics/json")
    def get_json_metrics():
        s = _s()
        if s.metrics_collector is None:
            return {"metrics": {}}
        return {"metrics": s.metrics_collector.to_json()}

    @router.get("/plugins")
    def list_plugins():
        s = _s()
        with s.engine.step_lock:
            return {
                "plugins": [
                    {"id": p.plugin_id, "active": p.active}
                    for p in s.engine._plugins
                ],
                "agents": [
                    {"id": a.agent_id}
                    for a in s.engine._agents
                ],
            }

    @router.delete("/plugins/{plugin_id}")
    def unload_plugin(plugin_id: str):
        s = _s()
        if s.hot_reload_manager is None:
            raise HTTPException(400, "HotReloadManager not configured")
        with s.engine.step_lock:
            ok = s.hot_reload_manager.unload_plugin(plugin_id)
        if not ok:
            raise HTTPException(404, f"Plugin '{plugin_id}' not found")
        return {"unloaded": plugin_id}

    @router.delete("/agents/{agent_id}")
    def unload_agent(agent_id: str):
        s = _s()
        if s.hot_reload_manager is None:
            raise HTTPException(400, "HotReloadManager not configured")
        with s.engine.step_lock:
            ok = s.hot_reload_manager.unload_agent(agent_id)
        if not ok:
            raise HTTPException(404, f"Agent '{agent_id}' not found")
        return {"unloaded": agent_id}

    return router
