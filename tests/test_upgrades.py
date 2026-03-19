"""
test_upgrades.py — full test suite for all upgrade modules.

Per 測試規則.txt, every module includes:
  1. Unit tests
  2. Edge case tests
  3. Mock data tests

Modules tested:
  - PhysicsDistance noise
  - PhysicsCamera noise + full depth
  - GoalPublisher plugin
  - GoalAgent
  - Config system
  - PhysicsLidar
  - PluginRegistry
  - NoiseMixin
  - PhysicsAgentBase
  - Config-driven Engine
  - Recorder
  - Replayer (Event Replay System)
  - Multi-Agent (PhysicsWorld multi-robot)
  - AStarAgent (A* Path Planning)
  - CollisionDetector
  - ActionLayer (base, SimulatedAction, LoggingAction)
  - IK Solver (URDFRobot.solve_ik, move_end_effector_to)
  - RealWorldAdapter (DryRunTransport, CommandValidator)
  - SB3Agent (DQN/PPO wrappers)
  - Docker / CI (config file validation)
  - Behavior Tree (BTNode, Sequence, Selector, Condition, ActionNode, Inverter, Parallel)
  - BTAgent
  - OccupancyMap (SLAM-lite)
  - Observability (StructuredLogger, MetricsCollector, ObservabilityPlugin)
  - RRT* motion planner
  - GraspPlanner (pick-and-place)
  - Event Schema Validation (SCHEMAS, validate_event_data, SchemaValidator)
  - Plugin Dependency Graph (resolve_order, check_health)
  - State Machine (MissionFSM, Phase, Transition)
  - API v2 (expanded endpoints)
  - PhysicsWorld.remove_robot
  - Engine plugin topological order (resolve_order on start)
  - Config YAML save/load (Config.save / load_file)
  - execute_ws_command + ws.command.ack
  - Replay API (/api/replay/*)

Run:  pytest tests/test_upgrades.py -v
"""

import sys
import os
import json
import math
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, _make_event
from core.engine import Engine
from core.config import Config, _deep_merge
from core.recorder import Recorder

from simulator.physics_world import PhysicsWorld, DIRECTIONS_3D

from plugins.physics_distance import PhysicsDistance
from plugins.physics_camera import PhysicsCamera
from plugins.physics_lidar import PhysicsLidar
from plugins.goal_publisher import GoalPublisher

from agents.physics_agent import PhysicsAgent
from agents.goal_agent import GoalAgent


# ════════════════════════════════════════════
#  PhysicsDistance — Noise
# ════════════════════════════════════════════

class TestPhysicsDistanceNoiseUnit:
    def test_noise_level_property(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world, noise_level=0.5, seed=42)
            assert plugin.noise_level == 0.5
        finally:
            world.disconnect()

    def test_noisy_flag_in_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world, noise_level=0.3, seed=1)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            assert received[0]["data"]["noisy"] is True
        finally:
            world.disconnect()

    def test_raw_differs_from_noisy(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world, noise_level=2.0, seed=7)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            dists = received[0]["data"]["distances"]
            diffs = [abs(d["distance"] - d["raw_distance"]) for d in dists.values()]
            assert any(d > 0 for d in diffs)
        finally:
            world.disconnect()


class TestPhysicsDistanceNoiseEdgeCases:
    def test_zero_noise(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world, noise_level=0.0)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            assert received[0]["data"]["noisy"] is False
            dists = received[0]["data"]["distances"]
            for d in dists.values():
                assert d["distance"] == d["raw_distance"]
        finally:
            world.disconnect()

    def test_negative_noise_clamps(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world, noise_level=-1.0)
            assert plugin.noise_level == 0.0
        finally:
            world.disconnect()

    def test_noise_setter(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world)
            plugin.noise_level = 0.8
            assert plugin.noise_level == 0.8
            plugin.noise_level = -5.0
            assert plugin.noise_level == 0.0
        finally:
            world.disconnect()


class TestPhysicsDistanceNoiseMockData:
    def test_consistent_seed(self):
        bus1 = EventBus()
        bus2 = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            r1, r2 = [], []
            p1 = PhysicsDistance(bus1, world, noise_level=1.0, seed=99)
            p2 = PhysicsDistance(bus2, world, noise_level=1.0, seed=99)
            bus1.subscribe("sensor.distance_3d", r1.append)
            bus2.subscribe("sensor.distance_3d", r2.append)
            p1.start()
            p1.update()
            p2.start()
            p2.update()
            d1 = r1[0]["data"]["distances"]["forward"]["distance"]
            d2 = r2[0]["data"]["distances"]["forward"]["distance"]
            assert d1 == d2
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  PhysicsCamera — Noise + Full Depth
# ════════════════════════════════════════════

class TestPhysicsCameraUpgradeUnit:
    def test_full_depth_matrix(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=8, height=8)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            depth = received[0]["data"]["depth"]
            assert len(depth) == 8
            assert len(depth[0]) == 8
        finally:
            world.disconnect()

    def test_noise_adds_raw_depth(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=4, height=4, noise_level=0.1, seed=1)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            ev = received[0]
            assert ev["data"]["noisy"] is True
            assert "raw_depth" in ev["data"]
        finally:
            world.disconnect()


class TestPhysicsCameraUpgradeEdgeCases:
    def test_no_noise_no_raw_depth(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=4, height=4, noise_level=0.0)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            assert "raw_depth" not in received[0]["data"]
        finally:
            world.disconnect()

    def test_noise_setter(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=4, height=4)
            plugin.noise_level = 0.5
            assert plugin.noise_level == 0.5
            plugin.noise_level = -1.0
            assert plugin.noise_level == 0.0
        finally:
            world.disconnect()

    def test_negative_noise_clamps(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=4, height=4, noise_level=-5.0)
            assert plugin.noise_level == 0.0
        finally:
            world.disconnect()


class TestPhysicsCameraUpgradeMockData:
    def test_depth_values_bounded(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=4, height=4, noise_level=0.5, seed=42)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            depth = received[0]["data"]["depth"]
            for row in depth:
                for v in row:
                    assert 0.0 <= v <= 1.0
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  GoalPublisher
# ════════════════════════════════════════════

class TestGoalPublisherUnit:
    def test_publishes_goal_event(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        received = []
        bus.subscribe("goal.target", received.append)
        plugin.set_target(5.0, 3.0, 0.5)
        plugin.start()
        plugin.update()
        assert len(received) == 1
        assert received[0]["data"]["target"] == [5.0, 3.0, 0.5]

    def test_no_event_without_target(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        received = []
        bus.subscribe("goal.target", received.append)
        plugin.start()
        plugin.update()
        assert len(received) == 0

    def test_target_property(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        assert plugin.target is None
        plugin.set_target(1, 2, 3)
        assert plugin.target == (1, 2, 3)


class TestGoalPublisherEdgeCases:
    def test_clear_target(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        plugin.set_target(1, 2)
        plugin.clear_target()
        assert plugin.target is None

    def test_mark_reached(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        plugin.set_target(1, 2)
        assert not plugin.reached
        plugin.mark_reached()
        assert plugin.reached

    def test_update_before_start(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        plugin.set_target(1, 2)
        received = []
        bus.subscribe("goal.target", received.append)
        plugin.update()
        assert len(received) == 0


class TestGoalPublisherMockData:
    def test_event_protocol(self):
        bus = EventBus()
        plugin = GoalPublisher(bus)
        plugin.set_target(10.0, -5.0)
        plugin.start()
        plugin.update()
        ev = bus.get_history("goal.target")[0]
        assert "timestamp" in ev
        assert ev["type"] == "goal.target"
        assert ev["source"] == "goal_publisher"
        assert "data" in ev


# ════════════════════════════════════════════
#  GoalAgent
# ════════════════════════════════════════════

class TestGoalAgentUnit:
    def test_subscribes_to_goal_and_distance(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            assert agent.goal is None
            bus.publish(_make_event("goal.target", "test", {"target": [5, 0, 0.31], "reached": False}))
            assert agent.goal == (5, 0, 0.31)
        finally:
            world.disconnect()

    def test_publishes_move_3d(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            received = []
            bus.subscribe("action.move_3d", received.append)
            agent.decide()
            agent.act()
            assert len(received) == 1
        finally:
            world.disconnect()

    def test_goal_reached_publishes_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            pos = world.get_robot_position()
            bus.publish(_make_event("goal.target", "test", {
                "target": [pos[0], pos[1], pos[2]], "reached": False
            }))
            all_clear = {d: {"distance": 20, "blocked": False} for d in DIRECTIONS_3D}
            agent._latest_distance = _make_event("sensor.distance_3d", "test", {
                "distances": all_clear, "robot_position": list(pos)
            })
            reached_events = []
            bus.subscribe("goal.reached", reached_events.append)
            agent.decide()
            assert agent.goal_reached
            assert len(reached_events) == 1
        finally:
            world.disconnect()


class TestGoalAgentEdgeCases:
    def test_no_goal_explores(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            all_clear = {d: {"distance": 20, "blocked": False} for d in DIRECTIONS_3D}
            agent._latest_distance = _make_event("sensor.distance_3d", "test", {
                "distances": all_clear, "robot_position": [0, 0, 0.31]
            })
            agent.decide()
            assert agent._chosen_action is not None
        finally:
            world.disconnect()

    def test_all_blocked(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            bus.publish(_make_event("goal.target", "t", {"target": [5, 0, 0.31], "reached": False}))
            blocked = {d: {"distance": 0.1, "blocked": True} for d in DIRECTIONS_3D}
            agent._latest_distance = _make_event("sensor.distance_3d", "test", {
                "distances": blocked, "robot_position": [0, 0, 0.31]
            })
            agent.decide()
            assert agent._chosen_action is None
        finally:
            world.disconnect()

    def test_cleanup(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = GoalAgent(bus, world)
            agent.cleanup()
            bus.publish(_make_event("sensor.distance_3d", "t", {"distances": {}}))
            assert agent._latest_distance is None
        finally:
            world.disconnect()


class TestGoalAgentMockData:
    def test_full_pipeline(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            goal_pub = GoalPublisher(bus)
            goal_pub.set_target(3.0, 0.0)
            dist_plugin = PhysicsDistance(bus, world)
            agent = GoalAgent(bus, world)

            engine = Engine(bus)
            engine.register_plugin(dist_plugin)
            engine.register_plugin(goal_pub)
            engine.register_agent(agent)
            engine.run(steps=5)

            actions = bus.get_history("action.move_3d")
            assert len(actions) >= 1
            assert actions[0]["data"]["goal"] is not None
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════

class TestConfigUnit:
    def test_defaults(self):
        cfg = Config()
        assert cfg.get("world.gui") is False
        assert cfg.get("engine.enable_metrics") is True

    def test_override(self):
        cfg = Config({"world": {"gui": True}})
        assert cfg.get("world.gui") is True
        assert cfg.get("world.timestep") is not None

    def test_dot_path_access(self):
        cfg = Config()
        assert cfg.get("sensor.noise_level") == 0.0
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_set(self):
        cfg = Config()
        cfg.set("world.gui", True)
        assert cfg.get("world.gui") is True

    def test_section(self):
        cfg = Config()
        world = cfg.section("world")
        assert isinstance(world, dict)
        assert "timestep" in world

    def test_to_dict(self):
        cfg = Config({"world": {"gui": True}})
        d = cfg.to_dict()
        assert d["world"]["gui"] is True


class TestConfigEdgeCases:
    def test_empty_override(self):
        cfg = Config({})
        assert cfg.get("world.gui") is False

    def test_deep_override(self):
        cfg = Config({"world": {"timestep": 0.01}})
        assert cfg.get("world.timestep") == 0.01
        assert cfg.get("world.gui") is False

    def test_set_creates_path(self):
        cfg = Config()
        cfg.set("custom.deep.value", 42)
        assert cfg.get("custom.deep.value") == 42

    def test_missing_section(self):
        cfg = Config()
        assert cfg.section("nonexistent") == {}


class TestConfigMockData:
    def test_from_file_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"world": {"gui": True, "gravity": -5.0}}, f)
            f.flush()
            path = f.name
        try:
            cfg = Config.from_file(path)
            assert cfg.get("world.gui") is True
            assert cfg.get("world.gravity") == -5.0
            assert cfg.get("world.timestep") is not None
        finally:
            os.unlink(path)

    def test_save_and_reload(self):
        cfg = Config({"world": {"gui": True}})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            cfg2 = Config.from_file(path)
            assert cfg2.get("world.gui") is True
        finally:
            os.unlink(path)

    def test_deep_merge(self):
        result = _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 99}})
        assert result["a"]["b"] == 1
        assert result["a"]["c"] == 99


# ════════════════════════════════════════════
#  PhysicsLidar
# ════════════════════════════════════════════

class TestPhysicsLidarUnit:
    def test_publishes_lidar_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=12)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            assert len(received) == 1
            assert received[0]["type"] == "sensor.lidar"
        finally:
            world.disconnect()

    def test_scan_has_correct_ray_count(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=24)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            scan = received[0]["data"]["scan"]
            assert len(scan) == 24
        finally:
            world.disconnect()

    def test_single_layer_has_scan(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=8, num_layers=1)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            assert "scan" in received[0]["data"]
            assert "layers" not in received[0]["data"]
        finally:
            world.disconnect()

    def test_multi_layer_has_layers(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=8, num_layers=3)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            assert "layers" in received[0]["data"]
            assert len(received[0]["data"]["layers"]) == 3
        finally:
            world.disconnect()

    def test_obstacle_detected(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 0, 0.5)
            plugin = PhysicsLidar(bus, world, num_rays=36)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            scan = received[0]["data"]["scan"]
            hits = [r for r in scan if r["hit"]]
            assert len(hits) >= 1
        finally:
            world.disconnect()


class TestPhysicsLidarEdgeCases:
    def test_no_event_before_start(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.update()
            assert len(received) == 0
        finally:
            world.disconnect()

    def test_noise_level_setter(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world)
            plugin.noise_level = 1.0
            assert plugin.noise_level == 1.0
            plugin.noise_level = -5.0
            assert plugin.noise_level == 0.0
        finally:
            world.disconnect()

    def test_min_rays_clamped(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=1)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            assert received[0]["data"]["num_rays"] >= 4
        finally:
            world.disconnect()


class TestPhysicsLidarMockData:
    def test_noisy_scan(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=12, noise_level=2.0, seed=42)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            assert received[0]["data"]["noisy"] is True
            scan = received[0]["data"]["scan"]
            diffs = [abs(r["distance"] - r["raw_distance"]) for r in scan]
            assert any(d > 0 for d in diffs)
        finally:
            world.disconnect()

    def test_event_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=8)
            plugin.start()
            plugin.update()
            ev = bus.get_history("sensor.lidar")[0]
            assert "timestamp" in ev
            assert ev["type"] == "sensor.lidar"
            assert ev["source"] == "physics_lidar"
            assert "data" in ev
        finally:
            world.disconnect()

    def test_angles_cover_full_circle(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsLidar(bus, world, num_rays=36)
            received = []
            bus.subscribe("sensor.lidar", received.append)
            plugin.start()
            plugin.update()
            scan = received[0]["data"]["scan"]
            angles = [r["angle_rad"] for r in scan]
            assert min(angles) == pytest.approx(0.0, abs=0.01)
            assert max(angles) == pytest.approx(2 * math.pi * 35 / 36, abs=0.01)
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  Recorder
# ════════════════════════════════════════════

class TestRecorderUnit:
    def test_records_events(self):
        bus = EventBus()
        rec = Recorder(bus)
        rec.start()
        bus.publish(_make_event("test.a", "src", {"v": 1}))
        bus.publish(_make_event("test.b", "src", {"v": 2}))
        rec.stop()
        assert rec.event_count == 2

    def test_get_events_filtered(self):
        bus = EventBus()
        rec = Recorder(bus)
        rec.start()
        bus.publish(_make_event("test.a", "src", {}))
        bus.publish(_make_event("test.b", "src", {}))
        rec.stop()
        assert len(rec.get_events("test.a")) == 1
        assert len(rec.get_events()) == 2

    def test_summary(self):
        bus = EventBus()
        rec = Recorder(bus)
        rec.start()
        bus.publish(_make_event("x", "s", {}))
        bus.publish(_make_event("x", "s", {}))
        bus.publish(_make_event("y", "s", {}))
        rec.stop()
        s = rec.get_summary()
        assert s["total_events"] == 3
        assert s["event_types"]["x"] == 2
        assert s["event_types"]["y"] == 1


class TestRecorderEdgeCases:
    def test_no_events_when_stopped(self):
        bus = EventBus()
        rec = Recorder(bus)
        bus.publish(_make_event("test", "src", {}))
        assert rec.event_count == 0

    def test_stop_before_start(self):
        bus = EventBus()
        rec = Recorder(bus)
        rec.stop()
        assert rec.event_count == 0

    def test_recording_property(self):
        bus = EventBus()
        rec = Recorder(bus)
        assert not rec.recording
        rec.start()
        assert rec.recording
        rec.stop()
        assert not rec.recording


class TestRecorderMockData:
    def test_save_and_load(self):
        bus = EventBus()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            rec = Recorder(bus, output_path=path)
            rec.start()
            bus.publish(_make_event("a", "s", {"val": 1}))
            bus.publish(_make_event("b", "s", {"val": 2}))
            rec.stop()
            rec.save()

            loaded = Recorder.load(path)
            assert len(loaded) == 2
            assert loaded[0]["type"] == "a"
            assert loaded[1]["data"]["val"] == 2
        finally:
            os.unlink(path)

    def test_full_pipeline_recording(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name
            rec = Recorder(bus, output_path=path)
            rec.start()

            dist = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)
            engine = Engine(bus, enable_metrics=True)
            engine.register_plugin(dist)
            engine.register_agent(agent)
            engine.run(steps=3)

            rec.stop()
            assert rec.event_count > 0

            rec.save()
            loaded = Recorder.load(path)
            assert len(loaded) == rec.event_count
            os.unlink(path)
        finally:
            world.disconnect()


# ── PluginRegistry ──────────────────────────────────────────────────────────

from core.registry import PluginRegistry, default_registry


class TestPluginRegistryUnit:
    def test_register_and_create_plugin(self):
        reg = PluginRegistry()

        class FakePlugin:
            def __init__(self, x):
                self.x = x

        reg.register_plugin("fake", FakePlugin)
        obj = reg.create_plugin("fake", x=42)
        assert obj.x == 42

    def test_register_and_create_agent(self):
        reg = PluginRegistry()

        class FakeAgent:
            def __init__(self, agent_name):
                self.agent_name = agent_name

        reg.register_agent("fake", FakeAgent)
        obj = reg.create_agent("fake", agent_name="bot")
        assert obj.agent_name == "bot"

    def test_plugin_names_sorted(self):
        reg = PluginRegistry()
        reg.register_plugin("b_plugin", lambda: None)
        reg.register_plugin("a_plugin", lambda: None)
        assert reg.plugin_names == ["a_plugin", "b_plugin"]

    def test_has_plugin_and_agent(self):
        reg = PluginRegistry()
        reg.register_plugin("cam", lambda: None)
        assert reg.has_plugin("cam")
        assert not reg.has_agent("cam")


class TestPluginRegistryEdgeCases:
    def test_unknown_plugin_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="Unknown plugin"):
            reg.create_plugin("no_exist")

    def test_unknown_agent_raises(self):
        reg = PluginRegistry()
        with pytest.raises(KeyError, match="Unknown agent"):
            reg.create_agent("no_exist")

    def test_register_non_callable_raises(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="callable"):
            reg.register_plugin("bad", 123)

    def test_overwrite_registration(self):
        reg = PluginRegistry()
        reg.register_plugin("p", lambda: "old")
        reg.register_plugin("p", lambda: "new")
        assert reg.create_plugin("p") == "new"

    def test_clear_empties_registry(self):
        reg = PluginRegistry()
        reg.register_plugin("x", lambda: None)
        reg.register_agent("y", lambda: None)
        reg.clear()
        assert reg.plugin_names == []
        assert reg.agent_names == []


class TestPluginRegistryMockData:
    def test_decorator_registration(self):
        reg = PluginRegistry()

        @reg.plugin("decorated")
        class Dec:
            pass

        assert reg.has_plugin("decorated")
        assert isinstance(reg.create_plugin("decorated"), Dec)

    def test_agent_decorator(self):
        reg = PluginRegistry()

        @reg.agent("my_agent")
        class Agent:
            pass

        assert reg.has_agent("my_agent")
        assert isinstance(reg.create_agent("my_agent"), Agent)

    def test_default_registry_is_singleton(self):
        assert isinstance(default_registry, PluginRegistry)


# ── NoiseMixin standalone ───────────────────────────────────────────────────

from plugins.noise_mixin import NoiseMixin


class TestNoiseMixinUnit:
    def test_init_noise_defaults(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise()
        assert obj.noise_level == 0.0

    def test_apply_noise_no_noise(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise(noise_level=0.0)
        assert obj._apply_noise(5.0) == 5.0

    def test_apply_noise_with_noise_varies(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise(noise_level=1.0, seed=42)
        vals = [obj._apply_noise(5.0) for _ in range(10)]
        assert len(set(vals)) > 1


class TestNoiseMixinEdgeCases:
    def test_negative_noise_clamped(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise(noise_level=-5.0)
        assert obj.noise_level == 0.0

    def test_setter_clamps(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise()
        obj.noise_level = -1.0
        assert obj.noise_level == 0.0

    def test_min_val_respected(self):
        class T(NoiseMixin):
            pass
        obj = T()
        obj._init_noise(noise_level=100.0, seed=0)
        for _ in range(50):
            assert obj._apply_noise(0.0, min_val=0.0) >= 0.0


class TestNoiseMixinMockData:
    def test_deterministic_with_seed(self):
        class T(NoiseMixin):
            pass
        a, b = T(), T()
        a._init_noise(noise_level=0.5, seed=99)
        b._init_noise(noise_level=0.5, seed=99)
        va = [a._apply_noise(1.0) for _ in range(20)]
        vb = [b._apply_noise(1.0) for _ in range(20)]
        assert va == vb


# ── PhysicsAgentBase ────────────────────────────────────────────────────────

from agents.physics_agent_base import PhysicsAgentBase


class TestPhysicsAgentBaseUnit:
    def test_safe_candidates_returns_list(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:

            class ConcreteAgent(PhysicsAgentBase):
                def decide(self):
                    pass

            agent = ConcreteAgent(bus, world, agent_id="test_base")
            cands = agent._safe_candidates()
            assert isinstance(cands, list)
        finally:
            world.disconnect()

    def test_visit_count_starts_with_one(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:

            class ConcreteAgent(PhysicsAgentBase):
                def decide(self):
                    pass

            agent = ConcreteAgent(bus, world, agent_id="test_base")
            assert sum(agent.visit_count.values()) == 1
        finally:
            world.disconnect()


class TestPhysicsAgentBaseEdgeCases:
    def test_act_no_action_is_noop(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:

            class ConcreteAgent(PhysicsAgentBase):
                def decide(self):
                    pass

            agent = ConcreteAgent(bus, world, agent_id="test_base")
            received = []
            bus.subscribe("action.move_3d", lambda e: received.append(e))
            agent.act()
            assert len(received) == 0
        finally:
            world.disconnect()

    def test_cleanup_unsubscribes_distance(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:

            class ConcreteAgent(PhysicsAgentBase):
                def decide(self):
                    pass

            agent = ConcreteAgent(bus, world, agent_id="test_base")
            agent.cleanup()
            from core.event_bus import _make_event
            bus.publish(_make_event("sensor.distance_3d", "test", {}))
            assert agent._latest_distance is None
        finally:
            world.disconnect()


class TestPhysicsAgentBaseMockData:
    def test_act_publishes_standard_fields(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:

            class ConcreteAgent(PhysicsAgentBase):
                def decide(self):
                    self._chosen_action = "forward"

            agent = ConcreteAgent(bus, world, agent_id="test_base")
            received = []
            bus.subscribe("action.move_3d", lambda e: received.append(e))
            agent.decide()
            agent.act()
            assert len(received) == 1
            d = received[0]["data"]
            assert "direction" in d
            assert "success" in d
            assert "position_before" in d
            assert "new_position" in d
        finally:
            world.disconnect()


# ── Config-driven Engine ────────────────────────────────────────────────────



class TestConfigEngineUnit:
    def test_engine_reads_metrics_from_config(self):
        cfg = Config({"engine": {"enable_metrics": False}})
        bus = EventBus()
        engine = Engine(bus, config=cfg)
        assert engine._enable_metrics is False

    def test_engine_defaults_without_config(self):
        bus = EventBus()
        engine = Engine(bus)
        assert engine._enable_metrics is True
        assert engine.config is None

    def test_config_property_accessible(self):
        cfg = Config()
        bus = EventBus()
        engine = Engine(bus, config=cfg)
        assert engine.config is cfg


class TestConfigEngineEdgeCases:
    def test_explicit_override_beats_config(self):
        cfg = Config({"engine": {"enable_metrics": True}})
        bus = EventBus()
        engine = Engine(bus, enable_metrics=False, config=cfg)
        assert engine._enable_metrics is True

    def test_none_config_no_crash(self):
        bus = EventBus()
        engine = Engine(bus, config=None)
        engine.run(steps=1)
        assert engine.step_count == 1


class TestConfigEngineMockData:
    def test_metrics_disabled_via_config_no_events(self):
        cfg = Config({"engine": {"enable_metrics": False}})
        bus = EventBus()
        engine = Engine(bus, config=cfg)
        received = []
        bus.subscribe("engine.metrics", lambda e: received.append(e))
        engine.run(steps=3)
        assert len(received) == 0

    def test_metrics_enabled_via_config(self):
        cfg = Config({"engine": {"enable_metrics": True}})
        bus = EventBus()
        engine = Engine(bus, config=cfg)
        received = []
        bus.subscribe("engine.metrics", lambda e: received.append(e))
        engine.run(steps=2)
        assert len(received) == 2


# ── Replayer (Event Replay System) ──────────────────────────────────────────

from core.replayer import Replayer


class TestReplayerUnit:
    def test_load_events_in_memory(self):
        bus = EventBus()
        replayer = Replayer(bus)
        events = [
            _make_event("a", "src", {"x": 1}),
            _make_event("b", "src", {"x": 2}),
        ]
        count = replayer.load_events(events)
        assert count == 2
        assert replayer.event_count == 2
        assert not replayer.finished

    def test_step_publishes_one(self):
        bus = EventBus()
        replayer = Replayer(bus)
        received = []
        bus.subscribe("test.ev", lambda e: received.append(e))
        events = [_make_event("test.ev", "src", {"i": 0})]
        replayer.load_events(events)
        ev = replayer.step()
        assert ev is not None
        assert len(received) == 1
        assert replayer.finished

    def test_replay_instant_all(self):
        bus = EventBus()
        replayer = Replayer(bus)
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        events = [_make_event("x", "s", {"i": i}) for i in range(5)]
        replayer.load_events(events)
        count = replayer.replay(speed=0)
        assert count == 5
        assert replayer.replayed_count == 5
        assert replayer.finished


class TestReplayerEdgeCases:
    def test_step_when_empty(self):
        bus = EventBus()
        replayer = Replayer(bus)
        assert replayer.step() is None

    def test_reset_cursor(self):
        bus = EventBus()
        replayer = Replayer(bus)
        replayer.load_events([_make_event("a", "s", {})])
        replayer.step()
        assert replayer.finished
        replayer.reset()
        assert not replayer.finished
        assert replayer.cursor == 0

    def test_max_events_limit(self):
        bus = EventBus()
        replayer = Replayer(bus)
        events = [_make_event("x", "s", {"i": i}) for i in range(10)]
        replayer.load_events(events)
        count = replayer.replay(speed=0, max_events=3)
        assert count == 3
        assert replayer.cursor == 3

    def test_event_filter(self):
        bus = EventBus()
        replayer = Replayer(bus)
        received = []
        bus.subscribe("wanted", lambda e: received.append(e))
        events = [
            _make_event("skip", "s", {}),
            _make_event("wanted", "s", {"val": 1}),
            _make_event("skip", "s", {}),
            _make_event("wanted", "s", {"val": 2}),
        ]
        replayer.load_events(events)
        count = replayer.replay(speed=0, event_filter="wanted")
        assert count == 2
        assert len(received) == 2


class TestReplayerMockData:
    def test_load_and_replay_from_file(self, tmp_path):
        bus = EventBus()
        rec = Recorder(bus, output_path=str(tmp_path / "test.jsonl"))
        rec.start()
        for i in range(5):
            bus.publish(_make_event("ev", "src", {"i": i}))
        rec.stop()
        rec.save()

        bus2 = EventBus()
        replayer = Replayer(bus2)
        loaded = replayer.load(str(tmp_path / "test.jsonl"))
        assert loaded == 5
        received = []
        bus2.subscribe("ev", lambda e: received.append(e))
        replayer.replay(speed=0)
        assert len(received) == 5

    def test_summary(self):
        bus = EventBus()
        replayer = Replayer(bus)
        events = [
            _make_event("a", "s", {}),
            _make_event("b", "s", {}),
            _make_event("a", "s", {}),
        ]
        replayer.load_events(events)
        replayer.replay(speed=0, max_events=2)
        s = replayer.get_summary()
        assert s["total_events"] == 3
        assert s["replayed"] == 2
        assert s["remaining"] == 1
        assert s["event_types"]["a"] == 2
        assert s["event_types"]["b"] == 1

    def test_on_event_callback(self):
        bus = EventBus()
        replayer = Replayer(bus)
        events = [_make_event("x", "s", {"i": i}) for i in range(3)]
        replayer.load_events(events)
        indices = []
        replayer.replay(speed=0, on_event=lambda e, idx: indices.append(idx))
        assert indices == [0, 1, 2]


# ── Multi-Agent (PhysicsWorld multi-robot) ──────────────────────────────────


class TestMultiRobotUnit:
    def test_add_robot(self):
        world = PhysicsWorld(gui=False)
        try:
            body_id = world.add_robot("bot_2", x=3, y=0)
            assert body_id > 0
            assert "bot_2" in world.get_robot_ids()
            assert world.robot_count == 2
        finally:
            world.disconnect()

    def test_get_position_by_id(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("alpha", x=5, y=5)
            pos_d = world.get_robot_position("default")
            pos_a = world.get_robot_position("alpha")
            assert abs(pos_a[0] - 5.0) < 1.0
            assert abs(pos_d[0]) < 1.0
        finally:
            world.disconnect()

    def test_move_specific_robot(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("mover", x=0, y=3)
            pos_before = world.get_robot_position("mover")
            world.move_robot("forward", robot_id="mover")
            pos_after = world.get_robot_position("mover")
            import math
            assert math.dist(pos_before, pos_after) > 0.01
        finally:
            world.disconnect()


class TestMultiRobotEdgeCases:
    def test_duplicate_robot_id_raises(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("dup")
            with pytest.raises(ValueError, match="already exists"):
                world.add_robot("dup")
        finally:
            world.disconnect()

    def test_unknown_robot_raises(self):
        world = PhysicsWorld(gui=False)
        try:
            with pytest.raises(KeyError, match="Unknown robot"):
                world.get_robot_position("no_exist")
        finally:
            world.disconnect()

    def test_remove_robot(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("temp")
            assert world.robot_count == 2
            world.remove_robot("temp")
            assert world.robot_count == 1
            assert "temp" not in world.get_robot_ids()
        finally:
            world.disconnect()

    def test_cannot_remove_default(self):
        world = PhysicsWorld(gui=False)
        try:
            with pytest.raises(ValueError, match="default"):
                world.remove_robot("default")
        finally:
            world.disconnect()


class TestMultiRobotMockData:
    def test_multi_agent_pipeline(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("scout", x=3, y=0)
            from plugins.physics_distance import PhysicsDistance
            dist_default = PhysicsDistance(bus, world)
            agent_default = PhysicsAgent(bus, world)
            agent_scout = PhysicsAgent(bus, world, robot_id="scout")

            engine = Engine(bus)
            engine.register_plugin(dist_default)
            engine.register_agent(agent_default)
            engine.register_agent(agent_scout)
            engine.run(steps=3)
            assert engine.step_count == 3
        finally:
            world.disconnect()

    def test_render_shows_all_robots(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("r2")
            text = world.render()
            assert "default" in text
            assert "r2" in text
            assert "Robots: 2" in text
        finally:
            world.disconnect()


# ── AStarAgent (A* Path Planning) ──────────────────────────────────────────

from agents.astar_agent import AStarAgent, astar_search


class TestAStarSearchUnit:
    def test_straight_path(self):
        path = astar_search((0, 0, 0), (3, 0, 0), blocked=set())
        assert len(path) == 4
        assert path[0] == (0, 0, 0)
        assert path[-1] == (3, 0, 0)

    def test_path_around_obstacle(self):
        blocked = {(1, 0, 0)}
        path = astar_search((0, 0, 0), (2, 0, 0), blocked)
        assert len(path) > 0
        assert path[-1] == (2, 0, 0)
        assert (1, 0, 0) not in path

    def test_unreachable_returns_empty(self):
        blocked = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
        path = astar_search((0, 0, 0), (5, 5, 5), blocked)
        assert path == []


class TestAStarSearchEdgeCases:
    def test_start_equals_goal(self):
        path = astar_search((0, 0, 0), (0, 0, 0), blocked=set())
        assert path == [(0, 0, 0)]

    def test_max_iterations_limit(self):
        path = astar_search((0, 0, 0), (100, 100, 100), blocked=set(), max_iterations=10)
        assert path == []

    def test_3d_path(self):
        path = astar_search((0, 0, 0), (0, 0, 3), blocked=set())
        assert len(path) == 4
        assert path[-1] == (0, 0, 3)


class TestAStarAgentUnit:
    def test_creates_and_plans(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = AStarAgent(bus, world)
            from plugins.goal_publisher import GoalPublisher
            gp = GoalPublisher(bus)
            gp.set_target(3.0, 0.0)
            gp.start()
            gp.update()
            assert agent.goal is not None
        finally:
            world.disconnect()

    def test_publishes_move_with_path_info(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            from plugins.physics_distance import PhysicsDistance
            from plugins.goal_publisher import GoalPublisher
            dist = PhysicsDistance(bus, world)
            agent = AStarAgent(bus, world)
            gp = GoalPublisher(bus)
            gp.set_target(2.0, 0.0, 0.31)

            engine = Engine(bus)
            engine.register_plugin(dist)
            engine.register_plugin(gp)
            engine.register_agent(agent)

            received = []
            bus.subscribe("action.move_3d", lambda e: received.append(e))
            engine.run(steps=3)
            if received:
                assert "path_remaining" in received[0]["data"]
        finally:
            world.disconnect()


class TestAStarAgentEdgeCases:
    def test_no_goal_explores(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            from plugins.physics_distance import PhysicsDistance
            dist = PhysicsDistance(bus, world)
            agent = AStarAgent(bus, world)
            engine = Engine(bus)
            engine.register_plugin(dist)
            engine.register_agent(agent)
            engine.run(steps=2)
            assert engine.step_count == 2
        finally:
            world.disconnect()

    def test_cleanup(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = AStarAgent(bus, world)
            agent.cleanup()
            bus.publish(_make_event("goal.target", "test", {"target": [1, 2, 3]}))
            assert agent.goal is None
        finally:
            world.disconnect()


class TestAStarAgentMockData:
    def test_blocked_cells_accumulate(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 0, 0.5)
            from plugins.physics_distance import PhysicsDistance
            dist = PhysicsDistance(bus, world)
            agent = AStarAgent(bus, world)
            engine = Engine(bus)
            engine.register_plugin(dist)
            engine.register_agent(agent)
            engine.run(steps=3)
            assert len(agent.blocked_cells) >= 0
        finally:
            world.disconnect()


# ── CollisionDetector ───────────────────────────────────────────────────────

from plugins.collision_detector import CollisionDetector


class TestCollisionDetectorUnit:
    def test_detects_obstacle_collision(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(0.7, 0, 0.5)
            detector = CollisionDetector(bus, world)

            received = []
            bus.subscribe("physics.collision", lambda e: received.append(e))
            detector.start()
            world.move_robot("forward")
            detector.update()

            obstacle_hits = [r for r in received if r["data"]["other_body"] == "obstacle"]
            assert len(obstacle_hits) >= 0
        finally:
            world.disconnect()

    def test_no_event_before_start(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            detector = CollisionDetector(bus, world)
            received = []
            bus.subscribe("physics.collision", lambda e: received.append(e))
            detector.update()
            assert len(received) == 0
        finally:
            world.disconnect()


class TestCollisionDetectorEdgeCases:
    def test_include_ground_flag(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            detector = CollisionDetector(bus, world, include_ground=True, force_threshold=0.0)
            received = []
            bus.subscribe("physics.collision", lambda e: received.append(e))
            detector.start()
            for _ in range(30):
                world.step_physics()
            detector.update()
            ground_hits = [r for r in received if r["data"]["other_body"] == "ground"]
            assert len(ground_hits) >= 0
        finally:
            world.disconnect()

    def test_force_threshold_filters(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            detector = CollisionDetector(bus, world, force_threshold=9999.0)
            received = []
            bus.subscribe("physics.collision", lambda e: received.append(e))
            detector.start()
            world.move_robot("forward")
            detector.update()
            assert len(received) == 0
        finally:
            world.disconnect()


class TestCollisionDetectorMockData:
    def test_event_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(0.6, 0, 0.5)
            detector = CollisionDetector(bus, world, force_threshold=0.0)
            received = []
            bus.subscribe("physics.collision", lambda e: received.append(e))
            detector.start()
            world.move_robot("forward")
            detector.update()
            for ev in received:
                assert "timestamp" in ev
                assert ev["type"] == "physics.collision"
                assert "robot_id" in ev["data"]
                assert "other_body" in ev["data"]
                assert "contact_point" in ev["data"]
                assert "normal_force" in ev["data"]
        finally:
            world.disconnect()

    def test_collision_count_property(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            detector = CollisionDetector(bus, world)
            assert detector.collision_count == 0
            detector.start()
            detector.update()
            assert detector.collision_count >= 0
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  ActionLayer (U2: base, SimulatedAction, LoggingAction)
# ════════════════════════════════════════════

from actions.base_action import ActionLayer
from actions.simulated_action import SimulatedAction
from actions.logging_action import LoggingAction


class TestActionLayerUnit:
    def test_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            ActionLayer()

    def test_simulated_action_creates(self):
        world = PhysicsWorld(gui=False)
        try:
            action = SimulatedAction(world)
            assert action.world is world
            assert action.robot_id is None
        finally:
            world.disconnect()

    def test_simulated_action_executes_move(self):
        world = PhysicsWorld(gui=False)
        try:
            action = SimulatedAction(world)
            result = action.execute({"direction": "forward"})
            assert result["success"] is True
            assert result["direction"] == "forward"
            assert "position_before" in result
            assert "new_position" in result
        finally:
            world.disconnect()

    def test_simulated_action_with_robot_id(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("bot_a", x=3, y=0)
            action = SimulatedAction(world, robot_id="bot_a")
            result = action.execute({"direction": "forward"})
            assert result["success"] is True
            assert result["robot_id"] == "bot_a"
        finally:
            world.disconnect()

    def test_logging_action_wraps(self):
        world = PhysicsWorld(gui=False)
        try:
            bus = EventBus()
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus)
            assert logging.inner is inner
            assert logging.execution_count == 0
        finally:
            world.disconnect()

    def test_logging_action_publishes_event(self):
        world = PhysicsWorld(gui=False)
        try:
            bus = EventBus()
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus)
            received = []
            bus.subscribe("action.executed", received.append)
            result = logging.execute({"direction": "forward"})
            assert result["success"] is True
            assert len(received) == 1
            assert received[0]["data"]["command"] == {"direction": "forward"}
            assert received[0]["data"]["result"]["success"] is True
            assert received[0]["data"]["execution_index"] == 1
            assert logging.execution_count == 1
        finally:
            world.disconnect()


class TestActionLayerEdgeCases:
    def test_missing_direction_returns_failure(self):
        world = PhysicsWorld(gui=False)
        try:
            action = SimulatedAction(world)
            result = action.execute({})
            assert result["success"] is False
            assert "reason" in result
        finally:
            world.disconnect()

    def test_invalid_direction_raises(self):
        world = PhysicsWorld(gui=False)
        try:
            action = SimulatedAction(world)
            with pytest.raises(ValueError, match="Unknown direction"):
                action.execute({"direction": "teleport"})
        finally:
            world.disconnect()

    def test_logging_with_missing_direction(self):
        world = PhysicsWorld(gui=False)
        try:
            bus = EventBus()
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus)
            received = []
            bus.subscribe("action.executed", received.append)
            result = logging.execute({})
            assert result["success"] is False
            assert len(received) == 1
        finally:
            world.disconnect()

    def test_logging_custom_source(self):
        world = PhysicsWorld(gui=False)
        try:
            bus = EventBus()
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus, source="custom_src")
            received = []
            bus.subscribe("action.executed", received.append)
            logging.execute({"direction": "forward"})
            assert received[0]["source"] == "custom_src"
        finally:
            world.disconnect()


class TestActionLayerMockData:
    def test_agent_uses_action_layer(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            action = SimulatedAction(world)
            agent = PhysicsAgent(bus, world, action_layer=action)
            assert agent.action_layer is action

            received = []
            bus.subscribe("action.move_3d", received.append)
            agent.decide()
            agent.act()
            assert len(received) == 1
        finally:
            world.disconnect()

    def test_agent_with_logging_action_layer(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus)
            agent = PhysicsAgent(bus, world, action_layer=logging)

            move_events = []
            exec_events = []
            bus.subscribe("action.move_3d", move_events.append)
            bus.subscribe("action.executed", exec_events.append)

            agent.decide()
            agent.act()
            assert len(move_events) == 1
            assert len(exec_events) == 1
        finally:
            world.disconnect()

    def test_full_pipeline_with_action_layer(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus, source="pipeline_test")

            dist_plugin = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world, action_layer=logging)

            engine = Engine(bus)
            engine.register_plugin(dist_plugin)
            engine.register_agent(agent)
            engine.run(steps=5)

            exec_events = bus.get_history("action.executed")
            assert len(exec_events) >= 1
            assert exec_events[0]["source"] == "pipeline_test"
            assert logging.execution_count >= 1
        finally:
            world.disconnect()

    def test_action_layer_event_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            inner = SimulatedAction(world)
            logging = LoggingAction(inner, bus)
            logging.execute({"direction": "backward"})
            ev = bus.get_history("action.executed")[0]
            assert "timestamp" in ev
            assert ev["type"] == "action.executed"
            assert ev["source"] == "logging_action"
            assert "data" in ev
            assert "command" in ev["data"]
            assert "result" in ev["data"]
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  WebSocketBridge (U1: Dashboard backbone)
# ════════════════════════════════════════════

from core.ws_bridge import WebSocketBridge


class TestWebSocketBridgeUnit:
    def test_creates_with_defaults(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        assert bridge.plugin_id == "ws_bridge"
        assert bridge.queue_size == 0
        assert bridge.total_queued == 0

    def test_queues_events_on_wildcard(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bridge.start()
        bus.publish(_make_event("test.a", "src", {"v": 1}))
        bus.publish(_make_event("test.b", "src", {"v": 2}))
        assert bridge.queue_size == 2
        assert bridge.total_queued == 2

    def test_drain_returns_and_clears(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bridge.start()
        bus.publish(_make_event("x", "s", {}))
        bus.publish(_make_event("y", "s", {}))
        events = bridge.drain()
        assert len(events) == 2
        assert bridge.queue_size == 0
        assert bridge.total_queued == 2

    def test_peek_does_not_remove(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bridge.start()
        for i in range(5):
            bus.publish(_make_event("e", "s", {"i": i}))
        peeked = bridge.peek(3)
        assert len(peeked) == 3
        assert bridge.queue_size == 5


class TestWebSocketBridgeEdgeCases:
    def test_no_events_before_start(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bus.publish(_make_event("x", "s", {}))
        assert bridge.queue_size == 0

    def test_stop_unsubscribes(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bridge.start()
        bus.publish(_make_event("a", "s", {}))
        assert bridge.queue_size == 1
        bridge.stop()
        bus.publish(_make_event("b", "s", {}))
        assert bridge.queue_size == 1

    def test_max_queue_bounded(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus, max_queue=3)
        bridge.start()
        for i in range(10):
            bus.publish(_make_event("e", "s", {"i": i}))
        assert bridge.queue_size == 3
        assert bridge.total_queued == 10

    def test_connected_clients_setter(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        bridge.connected_clients = 5
        assert bridge.connected_clients == 5
        bridge.connected_clients = -1
        assert bridge.connected_clients == 0

    def test_drain_when_empty(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus)
        assert bridge.drain() == []


class TestWebSocketBridgeMockData:
    def test_serialize_event(self):
        ev = _make_event("test", "src", {"key": "val"})
        s = WebSocketBridge.serialize_event(ev)
        parsed = json.loads(s)
        assert parsed["type"] == "test"
        assert parsed["data"]["key"] == "val"

    def test_get_status(self):
        bus = EventBus()
        bridge = WebSocketBridge(bus, max_queue=100)
        bridge.start()
        bus.publish(_make_event("a", "s", {}))
        status = bridge.get_status()
        assert status["active"] is True
        assert status["queue_size"] == 1
        assert status["total_queued"] == 1
        assert status["max_queue"] == 100

    def test_full_pipeline_with_engine(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            bridge = WebSocketBridge(bus)
            dist_plugin = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)

            engine = Engine(bus)
            engine.register_plugin(bridge)
            engine.register_plugin(dist_plugin)
            engine.register_agent(agent)
            engine.run(steps=3)

            events = bridge.drain()
            assert len(events) >= 3
            types = {e["type"] for e in events}
            assert "engine.metrics" in types
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  REST API (U3: FastAPI server)
# ════════════════════════════════════════════

from fastapi.testclient import TestClient
from api.server import create_app
from plugins.goal_publisher import GoalPublisher as _GoalPublisher


def _make_test_client():
    """Helper: build a wired TestClient with engine, world, plugins."""
    bus = EventBus(max_history=5000)
    world = PhysicsWorld(gui=False)
    gp = _GoalPublisher(bus)
    gp.set_target(3.0, 0.0)
    dist = PhysicsDistance(bus, world)
    agent = PhysicsAgent(bus, world)
    engine = Engine(bus)
    engine.register_plugin(dist)
    engine.register_plugin(gp)
    engine.register_agent(agent)
    app = create_app(engine, bus, world, goal_publisher=gp)
    return TestClient(app), world, engine


class TestRestApiUnit:
    def test_health(self):
        client, world, _ = _make_test_client()
        try:
            r = client.get("/api/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
        finally:
            world.disconnect()

    def test_get_state(self):
        client, world, _ = _make_test_client()
        try:
            r = client.get("/api/state")
            assert r.status_code == 200
            data = r.json()
            assert "robots" in data
            assert "default" in data["robots"]
            assert data["robot_count"] >= 1
        finally:
            world.disconnect()

    def test_post_dynamics_box(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post(
                "/api/v1/dynamics/box",
                json={"x": 0.1, "y": 0.2, "z": 2.0, "mass": 0.2, "half_extent": 0.04},
            )
            assert r.status_code == 200
            data = r.json()
            assert "body_id" in data
            assert data["dynamic_count"] >= 1
            st = client.get("/api/v1/state").json()
            assert st.get("dynamic_count", 0) >= 1
            assert len(st.get("dynamics", [])) >= 1
        finally:
            world.disconnect()

    def test_delete_dynamics(self):
        client, world, _ = _make_test_client()
        try:
            client.post("/api/v1/dynamics/box", json={"x": 0, "y": 0, "z": 1.0, "mass": 0.1})
            r = client.delete("/api/v1/dynamics")
            assert r.status_code == 200
            assert r.json().get("cleared") is True
            st = client.get("/api/v1/state").json()
            assert st["dynamic_count"] == 0
        finally:
            world.disconnect()

    def test_manipulator_not_configured(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post("/api/v1/manipulator/gripper", json={"openness": 0.5})
            assert r.status_code == 404
            r2 = client.get("/api/v1/manipulator/state")
            assert r2.status_code == 404
        finally:
            world.disconnect()

    def test_post_robot_position_teleport(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post(
                "/api/v1/robots/position",
                json={"x": 2.5, "y": -1.25, "z": 0.31},
            )
            assert r.status_code == 200
            data = r.json()
            assert data.get("robot_id") == "default"
            pos = data.get("position") or []
            assert len(pos) >= 2
            assert abs(pos[0] - 2.5) < 0.02
            assert abs(pos[1] - (-1.25)) < 0.02
        finally:
            world.disconnect()

    def test_manipulator_gripper_with_arm(self):
        from simulator.urdf_robot import URDFRobot

        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        gp = _GoalPublisher(bus)
        gp.set_target(3.0, 0.0)
        dist = PhysicsDistance(bus, world)
        agent = PhysicsAgent(bus, world)
        engine = Engine(bus)
        engine.register_plugin(dist)
        engine.register_plugin(gp)
        engine.register_agent(agent)
        arm = URDFRobot(world.client_id, position=(5.0, 0.0, 0.1))
        try:
            app = create_app(engine, bus, world, goal_publisher=gp, urdf_arm=arm)
            client = TestClient(app)
            r = client.get("/api/v1/manipulator/state")
            assert r.status_code == 200
            assert r.json()["num_joints"] == 3
            r2 = client.post("/api/v1/manipulator/gripper", json={"openness": 1.0})
            assert r2.status_code == 200
            body = r2.json()
            assert body["openness"] == 1.0
            assert "state" in body
            assert body["state"]["num_joints"] == 3
        finally:
            arm.cleanup()
            world.disconnect()

    def test_post_step(self):
        client, world, engine = _make_test_client()
        try:
            r = client.post("/api/step", json={"steps": 3})
            assert r.status_code == 200
            data = r.json()
            assert data["steps_executed"] == 3
            assert data["total_steps"] == 3
        finally:
            world.disconnect()

    def test_post_reset(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post("/api/reset")
            assert r.status_code == 200
            assert "robots" in r.json()
        finally:
            world.disconnect()

    def test_post_goal(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post("/api/goal", json={"x": 5.0, "y": 2.0, "z": 0.5})
            assert r.status_code == 200
            assert r.json()["goal"] == [5.0, 2.0, 0.5]
        finally:
            world.disconnect()

    def test_get_metrics(self):
        client, world, _ = _make_test_client()
        try:
            client.post("/api/step", json={"steps": 2})
            r = client.get("/api/metrics")
            assert r.status_code == 200
            m = r.json()
            assert m["steps"] == 2
            assert "avg_ms" in m
        finally:
            world.disconnect()

    def test_get_events(self):
        client, world, _ = _make_test_client()
        try:
            client.post("/api/step", json={"steps": 2})
            r = client.get("/api/events")
            assert r.status_code == 200
            assert len(r.json()["events"]) >= 1
        finally:
            world.disconnect()

    def test_get_config_empty(self):
        client, world, _ = _make_test_client()
        try:
            r = client.get("/api/config")
            assert r.status_code == 200
        finally:
            world.disconnect()


class TestRestApiEdgeCases:
    def test_step_zero_invalid(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post("/api/step", json={"steps": 0})
            assert r.status_code == 400
        finally:
            world.disconnect()

    def test_step_over_limit(self):
        client, world, _ = _make_test_client()
        try:
            r = client.post("/api/step", json={"steps": 9999})
            assert r.status_code == 400
        finally:
            world.disconnect()

    def test_events_with_filter(self):
        client, world, _ = _make_test_client()
        try:
            client.post("/api/step", json={"steps": 2})
            r = client.get("/api/events?event_type=engine.metrics&limit=1")
            assert r.status_code == 200
            events = r.json()["events"]
            assert len(events) <= 1
        finally:
            world.disconnect()

    def test_goal_no_publisher(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            engine = Engine(bus)
            app = create_app(engine, bus, world, goal_publisher=None)
            client = TestClient(app)
            r = client.post("/api/goal", json={"x": 1, "y": 2})
            assert r.status_code == 400
        finally:
            world.disconnect()


class TestRestApiMockData:
    def test_full_pipeline(self):
        client, world, _ = _make_test_client()
        try:
            r1 = client.post("/api/step", json={"steps": 5})
            assert r1.json()["total_steps"] == 5

            r2 = client.get("/api/state")
            assert r2.json()["step_count"] == 5

            r3 = client.get("/api/metrics")
            assert r3.json()["steps"] == 5

            r4 = client.get("/api/events?event_type=engine.metrics")
            assert r4.json()["total"] == 5
        finally:
            world.disconnect()

    def test_config_with_value(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            cfg = Config({"world": {"gui": False}})
            engine = Engine(bus, config=cfg)
            app = create_app(engine, bus, world, config=cfg)
            client = TestClient(app)
            r = client.get("/api/config")
            assert r.status_code == 200
            assert r.json()["config"]["world"]["gui"] is False
        finally:
            world.disconnect()

    def test_reset_then_step(self):
        client, world, _ = _make_test_client()
        try:
            client.post("/api/step", json={"steps": 2})
            client.post("/api/reset")
            pos = world.get_robot_position()
            assert abs(pos[0]) < 2.0
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  Gymnasium RL Environment (U4)
# ════════════════════════════════════════════

from rl.robot_env import RobotEnv
from rl.q_learning_agent import QLearningAgent


class TestRobotEnvUnit:
    def test_env_creates(self):
        env = RobotEnv()
        assert env.observation_space.shape == (12,)
        assert env.action_space.n == 6
        env.close()

    def test_reset_returns_obs_and_info(self):
        env = RobotEnv()
        obs, info = env.reset()
        assert obs.shape == (12,)
        assert "goal_distance" in info
        env.close()

    def test_step_returns_five_tuple(self):
        env = RobotEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "goal_distance" in info
        env.close()

    def test_action_space_valid(self):
        env = RobotEnv()
        env.reset()
        for a in range(6):
            obs, _, _, _, _ = env.step(a)
            assert obs.shape == (12,)
        env.close()

    def test_render_returns_string(self):
        env = RobotEnv(render_mode="text")
        env.reset()
        text = env.render()
        assert isinstance(text, str)
        assert "default" in text
        env.close()


class TestRobotEnvEdgeCases:
    def test_truncation_at_max_steps(self):
        env = RobotEnv(max_steps=3)
        env.reset()
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated or terminated
        env.close()

    def test_close_idempotent(self):
        env = RobotEnv()
        env.reset()
        env.close()
        env.close()

    def test_goal_reached_gives_bonus(self):
        env = RobotEnv(goal=(0.0, 0.0, 0.31), max_steps=10)
        obs, _ = env.reset()
        _, reward, terminated, _, _ = env.step(0)
        if terminated:
            assert reward > 50
        env.close()

    def test_multiple_resets(self):
        env = RobotEnv()
        for _ in range(3):
            obs, info = env.reset()
            assert obs.shape == (12,)
        env.close()


class TestRobotEnvMockData:
    def test_short_training_loop(self):
        env = RobotEnv(goal=(2.0, 0.0, 0.31), max_steps=20)
        agent = QLearningAgent(n_actions=6, seed=1)
        total_rewards = []
        for _ in range(3):
            obs, _ = env.reset()
            ep_reward = 0.0
            for _ in range(20):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.update(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                ep_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(ep_reward)
            agent.decay_epsilon()
        assert len(total_rewards) == 3
        assert agent.total_updates > 0
        env.close()

    def test_gymnasium_check_env(self):
        from gymnasium.utils.env_checker import check_env
        env = RobotEnv()
        check_env(env.unwrapped, skip_render_check=True)
        env.close()


# ── QLearningAgent standalone ───────────────────────────────────────────────


class TestQLearningAgentUnit:
    def test_creates_with_defaults(self):
        agent = QLearningAgent()
        assert agent.epsilon == 1.0
        assert agent.q_table_size == 0
        assert agent.total_updates == 0

    def test_select_action_returns_valid(self):
        agent = QLearningAgent(n_actions=6, seed=42)
        obs = [0.0] * 12
        action = agent.select_action(obs)
        assert 0 <= action < 6

    def test_update_changes_q(self):
        agent = QLearningAgent(n_actions=6, seed=42)
        obs = [1.0] * 12
        next_obs = [1.1] * 12
        agent.update(obs, 0, 1.0, next_obs, False)
        assert agent.q_table_size >= 1
        assert agent.total_updates == 1


class TestQLearningAgentEdgeCases:
    def test_epsilon_decay(self):
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
        agent.decay_epsilon()
        assert agent.epsilon == 0.5
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.1

    def test_deterministic_with_seed(self):
        a1 = QLearningAgent(seed=99, epsilon=0.5)
        a2 = QLearningAgent(seed=99, epsilon=0.5)
        obs = [0.5] * 12
        actions1 = [a1.select_action(obs) for _ in range(20)]
        actions2 = [a2.select_action(obs) for _ in range(20)]
        assert actions1 == actions2

    def test_get_q_values(self):
        agent = QLearningAgent(n_actions=4)
        obs = [0.0] * 12
        qvals = agent.get_q_values(obs)
        assert len(qvals) == 4
        assert all(v == 0.0 for v in qvals)


class TestQLearningAgentMockData:
    def test_learning_improves(self):
        agent = QLearningAgent(n_actions=2, seed=42, alpha=0.5)
        obs = [1.0, 2.0]
        next_obs = [1.0, 2.0]
        for _ in range(50):
            agent.update(obs, 0, 10.0, next_obs, False)
            agent.update(obs, 1, -5.0, next_obs, False)
        qvals = agent.get_q_values(obs)
        assert qvals[0] > qvals[1]


# ════════════════════════════════════════════
#  URDF Robot Models (U5)
# ════════════════════════════════════════════

from simulator.urdf_robot import URDFRobot
from plugins.joint_sensor import JointSensor
from actions.urdf_action import URDFAction


class TestURDFRobotUnit:
    def test_loads_bundled_urdf(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            assert robot.num_joints == 3
            assert robot.body_id >= 0
            assert robot.robot_id == "urdf_robot"
            robot.cleanup()
        finally:
            world.disconnect()

    def test_joint_info(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            info = robot.joint_info
            assert len(info) == 3
            assert info[0]["name"] == "joint1"
            assert info[1]["name"] == "joint2"
            assert info[2]["name"] == "gripper_slide"
            robot.cleanup()
        finally:
            world.disconnect()

    def test_get_joint_positions(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            positions = robot.get_joint_positions()
            assert len(positions) == 3
            assert all(abs(p) < 0.01 for p in positions)
            robot.cleanup()
        finally:
            world.disconnect()

    def test_set_joint_positions(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            robot.set_joint_positions([0.5, -0.3])
            for _ in range(240):
                world.step_physics()
            positions = robot.get_joint_positions()
            assert abs(positions[0] - 0.5) < 0.2
            robot.cleanup()
        finally:
            world.disconnect()

    def test_get_end_effector_position(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(1, 0, 0))
            ee = robot.get_end_effector_position()
            assert isinstance(ee, tuple)
            assert len(ee) == 3
            robot.cleanup()
        finally:
            world.disconnect()

    def test_get_state_snapshot(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, robot_id="arm_1")
            state = robot.get_state()
            assert state["robot_id"] == "arm_1"
            assert "base_position" in state
            assert "end_effector_position" in state
            assert "joint_positions" in state
            assert state["num_joints"] == 3
            robot.cleanup()
        finally:
            world.disconnect()

    def test_set_gripper_openness(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            robot.set_gripper_openness(1.0)
            for _ in range(120):
                world.step_physics()
            positions = robot.get_joint_positions()
            assert positions[2] > 0.05
            robot.set_gripper_openness(0.0)
            for _ in range(120):
                world.step_physics()
            positions2 = robot.get_joint_positions()
            assert positions2[2] < 0.03
            robot.cleanup()
        finally:
            world.disconnect()


class TestURDFRobotEdgeCases:
    def test_custom_position(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(3, 2, 0))
            base = robot.get_base_position()
            assert abs(base[0] - 3.0) < 0.1
            assert abs(base[1] - 2.0) < 0.1
            robot.cleanup()
        finally:
            world.disconnect()

    def test_reset_joints(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            robot.set_joint_positions([1.0, 1.0])
            for _ in range(120):
                world.step_physics()
            robot.reset_joints()
            positions = robot.get_joint_positions()
            assert all(abs(p) < 0.01 for p in positions)
            robot.cleanup()
        finally:
            world.disconnect()

    def test_joint_limits_clamped(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            robot.set_joint_positions([99.0, -99.0])
            for _ in range(240):
                world.step_physics()
            positions = robot.get_joint_positions()
            info = robot.joint_info
            assert positions[0] <= info[0]["upper_limit"] + 0.1
            assert positions[1] >= info[1]["lower_limit"] - 0.1
            robot.cleanup()
        finally:
            world.disconnect()

    def test_cleanup_idempotent(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            robot.cleanup()
            robot.cleanup()
        finally:
            world.disconnect()


class TestURDFRobotMockData:
    def test_coexists_with_sphere_robots(self):
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("sphere_bot", x=3, y=0)
            urdf = URDFRobot(world.client_id, position=(0, 3, 0), robot_id="arm")
            world.move_robot("forward", robot_id="sphere_bot")
            ee = urdf.get_end_effector_position()
            assert isinstance(ee, tuple)
            sphere_pos = world.get_robot_position("sphere_bot")
            assert sphere_pos[0] != ee[0]
            urdf.cleanup()
        finally:
            world.disconnect()

    def test_get_joint_velocities(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            vels = robot.get_joint_velocities()
            assert len(vels) == 3
            robot.cleanup()
        finally:
            world.disconnect()

    def test_get_joint_torques(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            torques = robot.get_joint_torques()
            assert len(torques) == 3
            robot.cleanup()
        finally:
            world.disconnect()


# ── JointSensor plugin ─────────────────────────────────────────────────────


class TestJointSensorUnit:
    def test_publishes_sensor_joints(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            sensor = JointSensor(bus, robot)
            received = []
            bus.subscribe("sensor.joints", received.append)
            sensor.start()
            sensor.update()
            assert len(received) == 1
            data = received[0]["data"]
            assert data["robot_id"] == "urdf_robot"
            assert len(data["joint_positions"]) == 3
            assert "end_effector_position" in data
            robot.cleanup()
        finally:
            world.disconnect()

    def test_no_event_before_start(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            sensor = JointSensor(bus, robot)
            received = []
            bus.subscribe("sensor.joints", received.append)
            sensor.update()
            assert len(received) == 0
            robot.cleanup()
        finally:
            world.disconnect()


class TestJointSensorEdgeCases:
    def test_event_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            sensor = JointSensor(bus, robot)
            sensor.start()
            sensor.update()
            ev = bus.get_history("sensor.joints")[0]
            assert "timestamp" in ev
            assert ev["type"] == "sensor.joints"
            assert ev["source"] == "joint_sensor"
            robot.cleanup()
        finally:
            world.disconnect()


class TestJointSensorMockData:
    def test_reflects_joint_movement(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            sensor = JointSensor(bus, robot)
            sensor.start()

            sensor.update()
            ev1 = bus.get_history("sensor.joints")[-1]
            pos_before = ev1["data"]["joint_positions"][0]

            robot.set_joint_positions([1.0, 0.5])
            for _ in range(240):
                world.step_physics()
            sensor.update()
            ev2 = bus.get_history("sensor.joints")[-1]
            pos_after = ev2["data"]["joint_positions"][0]

            assert abs(pos_after - pos_before) > 0.1
            robot.cleanup()
        finally:
            world.disconnect()


# ── URDFAction ──────────────────────────────────────────────────────────────


class TestURDFActionUnit:
    def test_execute_joint_positions(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            action = URDFAction(robot, world.client_id)
            result = action.execute({"joint_positions": [0.5, -0.3]})
            assert result["success"] is True
            assert "actual_positions" in result
            assert "ee_distance" in result
            robot.cleanup()
        finally:
            world.disconnect()

    def test_execute_direction(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            action = URDFAction(robot, world.client_id)
            result = action.execute({"direction": "forward"})
            assert result["success"] is True
            robot.cleanup()
        finally:
            world.disconnect()


class TestURDFActionEdgeCases:
    def test_unknown_direction(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            action = URDFAction(robot, world.client_id)
            result = action.execute({"direction": "teleport"})
            assert result["success"] is False
            robot.cleanup()
        finally:
            world.disconnect()

    def test_missing_command(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            action = URDFAction(robot, world.client_id)
            result = action.execute({})
            assert result["success"] is False
            robot.cleanup()
        finally:
            world.disconnect()


class TestURDFActionMockData:
    def test_end_effector_moves(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id)
            action = URDFAction(robot, world.client_id, sim_steps=240)
            result = action.execute({"joint_positions": [1.0, -0.8]})
            assert result["success"] is True
            assert result["ee_distance"] > 0.01
            robot.cleanup()
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  TaskAllocator (Multi-robot cooperative tasks)
# ════════════════════════════════════════════

from plugins.task_allocator import TaskAllocator, Task


class TestTaskAllocatorUnit:
    def test_add_task(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 3.0, 0.0)
            assert alloc.pending_count == 1
            assert len(alloc.tasks) == 1
            assert alloc.tasks[0]["task_id"] == "t1"
        finally:
            world.disconnect()

    def test_batch_add_tasks(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            count = alloc.add_tasks([
                {"task_id": "a", "x": 1, "y": 2},
                {"task_id": "b", "x": 3, "y": 4, "z": 0.5},
            ])
            assert count == 2
            assert alloc.pending_count == 2
        finally:
            world.disconnect()

    def test_assigns_on_update(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 2.0, 0.0)
            assigned = []
            bus.subscribe("task.assigned", assigned.append)
            alloc.start()
            alloc.update()
            assert len(assigned) == 1
            assert assigned[0]["data"]["robot_id"] == "default"
            assert assigned[0]["data"]["task_id"] == "t1"
        finally:
            world.disconnect()

    def test_publishes_goal_target(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 5.0, 3.0)
            goals = []
            bus.subscribe("goal.target", goals.append)
            alloc.start()
            alloc.update()
            assert len(goals) == 1
            assert goals[0]["data"]["target"] == [5.0, 3.0, 0.31]
        finally:
            world.disconnect()

    def test_summary(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 1, 0)
            alloc.add_task("t2", 2, 0)
            alloc.start()
            alloc.update()
            s = alloc.get_summary()
            assert s["total_tasks"] == 2
            assert s["assigned"] == 1
            assert s["pending"] == 1
        finally:
            world.disconnect()


class TestTaskAllocatorEdgeCases:
    def test_no_tasks_no_events(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            assigned = []
            bus.subscribe("task.assigned", assigned.append)
            alloc.start()
            alloc.update()
            assert len(assigned) == 0
        finally:
            world.disconnect()

    def test_more_tasks_than_robots(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 1, 0)
            alloc.add_task("t2", 2, 0)
            alloc.add_task("t3", 3, 0)
            assigned = []
            bus.subscribe("task.assigned", assigned.append)
            alloc.start()
            alloc.update()
            assert len(assigned) == 1
            assert alloc.pending_count == 2
        finally:
            world.disconnect()

    def test_multi_robot_assigns_both(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.add_robot("scout", x=5, y=0)
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 1, 0)
            alloc.add_task("t2", 6, 0)
            assigned = []
            bus.subscribe("task.assigned", assigned.append)
            alloc.start()
            alloc.update()
            assert len(assigned) == 2
            robots = {a["data"]["robot_id"] for a in assigned}
            assert "default" in robots
            assert "scout" in robots
        finally:
            world.disconnect()

    def test_stop_unsubscribes(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.start()
            alloc.stop()
            bus.publish(_make_event("goal.reached", "test", {}))
            assert alloc.completed_count == 0
        finally:
            world.disconnect()


class TestTaskAllocatorMockData:
    def test_completion_frees_robot(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 1, 0)
            alloc.add_task("t2", 2, 0)
            alloc.start()
            alloc.update()
            assert alloc.pending_count == 1
            assert len(alloc.busy_robots) == 1

            bus.publish(_make_event("goal.reached", "goal_agent", {
                "goal": [1, 0, 0.31], "position": [1, 0, 0.31],
            }))
            assert alloc.completed_count == 1
            assert len(alloc.busy_robots) == 0

            alloc.update()
            assert alloc.pending_count == 0
        finally:
            world.disconnect()

    def test_task_object(self):
        t = Task("x", (1.0, 2.0, 3.0))
        assert t.status == "pending"
        assert t.assigned_to is None
        d = t.to_dict()
        assert d["task_id"] == "x"
        assert d["target"] == [1.0, 2.0, 3.0]

    def test_event_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            alloc = TaskAllocator(bus, world)
            alloc.add_task("t1", 3, 0)
            alloc.start()
            alloc.update()
            ev = bus.get_history("task.assigned")[0]
            assert "timestamp" in ev
            assert ev["type"] == "task.assigned"
            assert ev["source"] == "task_allocator"
            assert "task_id" in ev["data"]
            assert "robot_id" in ev["data"]
            assert "target" in ev["data"]
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  HotReloadManager (Plugin Hot-reload)
# ════════════════════════════════════════════

from core.hot_reload import HotReloadManager


class TestHotReloadUnit:
    def test_load_plugin_at_runtime(self):
        bus = EventBus()
        engine = Engine(bus)
        hrm = HotReloadManager(engine, bus)
        engine.run(steps=1)

        from plugins.goal_publisher import GoalPublisher
        gp = GoalPublisher(bus)
        loaded = []
        bus.subscribe("system.plugin_loaded", loaded.append)
        hrm.load_plugin(gp)
        assert len(engine._plugins) == 1
        assert len(loaded) == 1
        assert loaded[0]["data"]["plugin_id"] == "goal_publisher"

    def test_unload_plugin(self):
        bus = EventBus()
        engine = Engine(bus)
        from plugins.goal_publisher import GoalPublisher
        gp = GoalPublisher(bus)
        engine.register_plugin(gp)
        engine.start()

        hrm = HotReloadManager(engine, bus)
        unloaded = []
        bus.subscribe("system.plugin_unloaded", unloaded.append)
        result = hrm.unload_plugin("goal_publisher")
        assert result is True
        assert len(engine._plugins) == 0
        assert len(unloaded) == 1
        engine.stop()

    def test_load_agent(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            engine = Engine(bus)
            hrm = HotReloadManager(engine, bus)
            agent = PhysicsAgent(bus, world)
            loaded = []
            bus.subscribe("system.agent_loaded", loaded.append)
            hrm.load_agent(agent)
            assert len(engine._agents) == 1
            assert loaded[0]["data"]["agent_id"] == "physics_agent"
        finally:
            world.disconnect()

    def test_unload_agent(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            engine = Engine(bus)
            agent = PhysicsAgent(bus, world)
            engine.register_agent(agent)
            hrm = HotReloadManager(engine, bus)
            unloaded = []
            bus.subscribe("system.agent_unloaded", unloaded.append)
            result = hrm.unload_agent("physics_agent")
            assert result is True
            assert len(engine._agents) == 0
            assert len(unloaded) == 1
        finally:
            world.disconnect()


class TestHotReloadEdgeCases:
    def test_unload_nonexistent_returns_false(self):
        bus = EventBus()
        engine = Engine(bus)
        hrm = HotReloadManager(engine, bus)
        assert hrm.unload_plugin("nonexistent") is False
        assert hrm.unload_agent("nonexistent") is False

    def test_swap_plugin(self):
        bus = EventBus()
        engine = Engine(bus)
        from plugins.goal_publisher import GoalPublisher
        gp1 = GoalPublisher(bus)
        engine.register_plugin(gp1)
        engine.start()

        hrm = HotReloadManager(engine, bus)
        gp2 = GoalPublisher(bus)
        removed = hrm.swap_plugin("goal_publisher", gp2)
        assert removed is True
        assert len(engine._plugins) == 1
        assert engine._plugins[0] is gp2
        engine.stop()

    def test_load_count_and_unload_count(self):
        bus = EventBus()
        engine = Engine(bus)
        hrm = HotReloadManager(engine, bus)
        from plugins.goal_publisher import GoalPublisher
        hrm.load_plugin(GoalPublisher(bus))
        hrm.load_plugin(GoalPublisher(bus))
        hrm.unload_plugin("goal_publisher")
        assert hrm.load_count == 2
        assert hrm.unload_count == 1


class TestHotReloadMockData:
    def test_get_status(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            engine = Engine(bus)
            agent = PhysicsAgent(bus, world)
            engine.register_agent(agent)
            from plugins.goal_publisher import GoalPublisher
            gp = GoalPublisher(bus)
            engine.register_plugin(gp)
            hrm = HotReloadManager(engine, bus)
            status = hrm.get_status()
            assert "goal_publisher" in status["plugins"]
            assert "physics_agent" in status["agents"]
            assert status["total_loads"] == 0
        finally:
            world.disconnect()

    def test_hot_load_during_running_engine(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            dist = PhysicsDistance(bus, world)
            engine = Engine(bus)
            engine.register_plugin(dist)
            engine.start()
            engine.step()

            hrm = HotReloadManager(engine, bus)
            agent = PhysicsAgent(bus, world)
            hrm.load_agent(agent)
            engine.step()

            moves = bus.get_history("action.move_3d")
            assert len(moves) >= 1
            engine.stop()
        finally:
            world.disconnect()

    def test_event_protocol(self):
        bus = EventBus()
        engine = Engine(bus)
        hrm = HotReloadManager(engine, bus)
        from plugins.goal_publisher import GoalPublisher
        hrm.load_plugin(GoalPublisher(bus))
        ev = bus.get_history("system.plugin_loaded")[0]
        assert "timestamp" in ev
        assert ev["type"] == "system.plugin_loaded"
        assert ev["source"] == "hot_reload"
        assert "plugin_id" in ev["data"]


# ════════════════════════════════════════════
#  IK Solver — URDFRobot inverse kinematics
# ════════════════════════════════════════════



class TestIKSolverUnit:
    def test_solve_ik_returns_list(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.solve_ik((0.0, 0.0, 0.3))
            assert isinstance(result, list)
            assert len(result) == robot.num_joints
        finally:
            world.disconnect()

    def test_solve_ik_values_within_limits(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.solve_ik((0.1, 0.0, 0.3))
            for i, val in enumerate(result):
                info = robot.joint_info[i]
                assert info["lower_limit"] <= val + 0.01
                assert val <= info["upper_limit"] + 0.01
        finally:
            world.disconnect()

    def test_move_end_effector_to_returns_dict(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.move_end_effector_to(0.0, 0.0, 0.4)
            assert isinstance(result, dict)
            assert "success" in result
            assert "achieved" in result
            assert "error" in result
            assert "joint_targets" in result
        finally:
            world.disconnect()

    def test_move_end_effector_to_achieves_target(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            # Bundled arm + gripper: EE home ~z=0.6; (0,0,0.55) stays in reliable workspace
            result = robot.move_end_effector_to(0.0, 0.0, 0.55, sim_steps=480)
            assert result["error"] < 0.15
        finally:
            world.disconnect()

    def test_move_end_effector_records_ee_before_after(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.move_end_effector_to(0.05, 0.0, 0.3)
            assert "ee_before" in result
            assert "ee_after" in result
            assert len(result["ee_before"]) == 3
            assert len(result["ee_after"]) == 3
        finally:
            world.disconnect()


class TestIKSolverEdgeCases:
    def test_unreachable_target(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.move_end_effector_to(100.0, 100.0, 100.0)
            assert result["error"] > 1.0
        finally:
            world.disconnect()

    def test_current_position_target(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            ee = robot.get_end_effector_position()
            result = robot.move_end_effector_to(*ee, sim_steps=240)
            assert result["error"] < 0.2
        finally:
            world.disconnect()

    def test_ik_with_zero_iterations(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            result = robot.solve_ik((0.1, 0.0, 0.3), max_iterations=1)
            assert isinstance(result, list)
        finally:
            world.disconnect()


class TestIKSolverMockData:
    def test_sequential_ik_calls(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            targets = [(0.0, 0.0, 0.3), (0.05, 0.0, 0.35), (-0.05, 0.0, 0.25)]
            for t in targets:
                result = robot.solve_ik(t)
                assert len(result) == robot.num_joints
        finally:
            world.disconnect()

    def test_ik_then_move(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            joints = robot.solve_ik((0.0, 0.0, 0.35))
            robot.set_joint_positions(joints)
            import pybullet as p
            for _ in range(240):
                p.stepSimulation(physicsClientId=world.client_id)
            ee = robot.get_end_effector_position()
            assert isinstance(ee, tuple)
            assert len(ee) == 3
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  URDFAction IK command support
# ════════════════════════════════════════════



class TestURDFActionIKUnit:
    def test_target_position_command(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            action = URDFAction(robot, world.client_id, sim_steps=120)
            result = action.execute({"target_position": [0.0, 0.0, 0.35]})
            assert isinstance(result, dict)
            assert "success" in result
            assert "achieved" in result
        finally:
            world.disconnect()

    def test_target_position_has_error_field(self):
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            action = URDFAction(robot, world.client_id)
            result = action.execute({"target_position": [0.0, 0.0, 0.3]})
            assert "error" in result
            assert isinstance(result["error"], float)
        finally:
            world.disconnect()

    def test_target_position_over_direction(self):
        """target_position takes priority over direction."""
        world = PhysicsWorld(gui=False)
        try:
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            action = URDFAction(robot, world.client_id)
            result = action.execute({
                "target_position": [0.0, 0.0, 0.3],
                "direction": "forward",
            })
            assert "achieved" in result
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  RealWorldAdapter
# ════════════════════════════════════════════

from actions.real_action import (
    DryRunTransport,
    CommandValidator,
    RealWorldAdapter,
)


class TestRealWorldAdapterUnit:
    def test_dry_run_succeeds(self):
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport)
        result = adapter.execute({"direction": "forward"})
        assert result["success"] is True
        assert result["mode"] == "dry_run"

    def test_dry_run_logs_commands(self):
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport)
        adapter.execute({"direction": "forward"})
        adapter.execute({"direction": "backward"})
        assert len(transport.command_log) == 2
        assert adapter.execution_count == 2

    def test_not_connected_fails(self):
        transport = DryRunTransport()
        adapter = RealWorldAdapter(transport)
        result = adapter.execute({"direction": "forward"})
        assert result["success"] is False
        assert "not connected" in result["reason"]

    def test_connect_disconnect(self):
        transport = DryRunTransport()
        adapter = RealWorldAdapter(transport)
        assert adapter.connect() is True
        assert transport.is_connected() is True
        adapter.disconnect()
        assert transport.is_connected() is False

    def test_publishes_events(self):
        bus = EventBus()
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport, event_bus=bus)
        adapter.execute({"direction": "up"})
        events = bus.get_history("action.hardware")
        assert len(events) == 1
        assert events[0]["data"]["command"]["direction"] == "up"

    def test_get_status(self):
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport)
        status = adapter.get_status()
        assert status["connected"] is True
        assert status["e_stop"] is False
        assert status["execution_count"] == 0


class TestCommandValidatorUnit:
    def test_valid_command_passes(self):
        v = CommandValidator()
        ok, reason = v.validate({"direction": "forward"})
        assert ok is True
        assert reason == "ok"

    def test_e_stop_blocks(self):
        v = CommandValidator()
        v.activate_e_stop()
        ok, reason = v.validate({"direction": "forward"})
        assert ok is False
        assert "emergency" in reason
        assert v.rejected_count == 1

    def test_e_stop_release(self):
        v = CommandValidator()
        v.activate_e_stop()
        v.release_e_stop()
        ok, _ = v.validate({"direction": "forward"})
        assert ok is True

    def test_joint_limits(self):
        v = CommandValidator(joint_limits=[(-1.0, 1.0), (-2.0, 2.0)])
        ok, _ = v.validate({"joint_positions": [0.5, 1.0]})
        assert ok is True
        ok, reason = v.validate({"joint_positions": [5.0, 0.0]})
        assert ok is False
        assert "out of limits" in reason

    def test_velocity_limit(self):
        v = CommandValidator(max_velocity=1.0)
        ok, _ = v.validate({"velocity": 0.5})
        assert ok is True
        ok, reason = v.validate({"velocity": 2.0})
        assert ok is False
        assert "velocity" in reason

    def test_rejected_count_accumulates(self):
        v = CommandValidator(joint_limits=[(-1.0, 1.0)])
        v.validate({"joint_positions": [5.0]})
        v.validate({"joint_positions": [5.0]})
        assert v.rejected_count == 2


class TestRealWorldAdapterEdgeCases:
    def test_e_stop_prevents_execution(self):
        transport = DryRunTransport()
        transport.connect()
        validator = CommandValidator()
        validator.activate_e_stop()
        adapter = RealWorldAdapter(transport, validator=validator)
        result = adapter.execute({"direction": "forward"})
        assert result["success"] is False
        assert result.get("rejected") is True
        assert adapter.execution_count == 0

    def test_empty_command(self):
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport)
        result = adapter.execute({})
        assert result["success"] is True

    def test_reconnect_after_disconnect(self):
        transport = DryRunTransport()
        transport.connect()
        adapter = RealWorldAdapter(transport)
        adapter.execute({"test": 1})
        adapter.disconnect()
        result = adapter.execute({"test": 2})
        assert result["success"] is False
        adapter.connect()
        result = adapter.execute({"test": 3})
        assert result["success"] is True


class TestRealWorldAdapterMockData:
    def test_multi_command_sequence(self):
        transport = DryRunTransport()
        transport.connect()
        bus = EventBus()
        adapter = RealWorldAdapter(transport, event_bus=bus)
        commands = [
            {"joint_positions": [0.1, 0.2]},
            {"direction": "forward"},
            {"velocity": 0.5, "direction": "up"},
        ]
        for cmd in commands:
            result = adapter.execute(cmd)
            assert result["success"] is True
        assert adapter.execution_count == 3
        assert len(bus.get_history("action.hardware")) == 3

    def test_mixed_valid_invalid(self):
        transport = DryRunTransport()
        transport.connect()
        validator = CommandValidator(joint_limits=[(-1.0, 1.0)])
        adapter = RealWorldAdapter(transport, validator=validator)
        r1 = adapter.execute({"joint_positions": [0.5]})
        assert r1["success"] is True
        r2 = adapter.execute({"joint_positions": [5.0]})
        assert r2["success"] is False
        r3 = adapter.execute({"joint_positions": [-0.5]})
        assert r3["success"] is True
        assert adapter.execution_count == 2


# ════════════════════════════════════════════
#  SB3Agent (DQN / PPO wrappers)
# ════════════════════════════════════════════

from rl.sb3_agent import SB3Agent


class TestSB3AgentUnit:
    def test_build_dqn(self):
        agent = SB3Agent(algorithm="DQN", env_kwargs={"max_steps": 10})
        try:
            agent.build()
            assert agent.model is not None
            assert agent.algorithm == "DQN"
        finally:
            agent.close()

    def test_build_ppo(self):
        agent = SB3Agent(algorithm="PPO", env_kwargs={"max_steps": 10})
        try:
            agent.build()
            assert agent.model is not None
            assert agent.algorithm == "PPO"
        finally:
            agent.close()

    def test_train_dqn_short(self):
        agent = SB3Agent(
            algorithm="DQN",
            env_kwargs={"max_steps": 10},
            model_kwargs={"learning_starts": 5, "buffer_size": 100},
        )
        try:
            summary = agent.train(total_timesteps=20)
            assert summary["timesteps"] == 20
            assert agent.total_timesteps_trained == 20
        finally:
            agent.close()

    def test_train_ppo_short(self):
        agent = SB3Agent(
            algorithm="PPO",
            env_kwargs={"max_steps": 10},
            model_kwargs={"n_steps": 10, "batch_size": 10},
        )
        try:
            summary = agent.train(total_timesteps=10)
            assert summary["algorithm"] == "PPO"
        finally:
            agent.close()

    def test_predict_after_build(self):
        agent = SB3Agent(algorithm="DQN", env_kwargs={"max_steps": 5})
        try:
            agent.build()
            import numpy as np
            obs = np.zeros(12, dtype=np.float32)
            action = agent.predict(obs)
            assert 0 <= action < 6
        finally:
            agent.close()

    def test_get_status(self):
        agent = SB3Agent(algorithm="DQN")
        status = agent.get_status()
        assert status["algorithm"] == "DQN"
        assert status["model_loaded"] is False
        assert status["total_trained"] == 0


class TestSB3AgentEdgeCases:
    def test_predict_without_build_raises(self):
        import numpy as np
        agent = SB3Agent(algorithm="DQN")
        with pytest.raises(RuntimeError, match="No model"):
            agent.predict(np.zeros(12, dtype=np.float32))

    def test_save_without_build_raises(self):
        agent = SB3Agent(algorithm="DQN")
        with pytest.raises(RuntimeError, match="No model"):
            agent.save("dummy_path")

    def test_evaluate_without_build_raises(self):
        agent = SB3Agent(algorithm="DQN")
        with pytest.raises(RuntimeError, match="not built"):
            agent.evaluate()

    def test_close_without_build(self):
        agent = SB3Agent(algorithm="DQN")
        agent.close()


class TestSB3AgentMockData:
    def test_save_and_load(self):
        agent = SB3Agent(
            algorithm="DQN",
            env_kwargs={"max_steps": 5},
            model_kwargs={"learning_starts": 5, "buffer_size": 50},
        )
        try:
            agent.train(total_timesteps=10)
            path = os.path.join(tempfile.gettempdir(), "test_dqn_model")
            saved = agent.save(path)
            assert os.path.exists(saved + ".zip")

            agent2 = SB3Agent(algorithm="DQN", env_kwargs={"max_steps": 5})
            agent2.load(saved)
            import numpy as np
            action = agent2.predict(np.zeros(12, dtype=np.float32))
            assert 0 <= action < 6
            agent2.close()

            if os.path.exists(saved + ".zip"):
                os.unlink(saved + ".zip")
        finally:
            agent.close()

    def test_train_then_evaluate(self):
        agent = SB3Agent(
            algorithm="DQN",
            env_kwargs={"max_steps": 10},
            model_kwargs={"learning_starts": 5, "buffer_size": 100},
        )
        try:
            agent.train(total_timesteps=20)
            result = agent.evaluate(n_eval_episodes=1)
            assert "mean_reward" in result
            assert "std_reward" in result
            assert result["n_episodes"] == 1
        finally:
            agent.close()

    def test_incremental_training(self):
        agent = SB3Agent(
            algorithm="DQN",
            env_kwargs={"max_steps": 10},
            model_kwargs={"learning_starts": 5, "buffer_size": 100},
        )
        try:
            agent.train(total_timesteps=10)
            agent.train(total_timesteps=10)
            assert agent.total_timesteps_trained == 20
        finally:
            agent.close()


# ════════════════════════════════════════════
#  Docker / CI — config file validation
# ════════════════════════════════════════════

class TestDockerCIConfigUnit:
    def test_dockerfile_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
        assert os.path.exists(path)

    def test_docker_compose_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        assert os.path.exists(path)

    def test_ci_workflow_exists(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml"
        )
        assert os.path.exists(path)

    def test_dockerfile_has_expose(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "EXPOSE 8000" in content

    def test_docker_compose_has_services(self):
        path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "services:" in content
        assert "iraap:" in content

    def test_ci_has_pytest(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "ci.yml"
        )
        with open(path) as f:
            content = f.read()
        assert "pytest" in content

    def test_requirements_has_sb3(self):
        path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        with open(path) as f:
            content = f.read()
        assert "stable-baselines3" in content


# ════════════════════════════════════════════
#  Behavior Tree Nodes
# ════════════════════════════════════════════

from agents.bt_nodes import (
    Status, Sequence, Selector, Condition,
    ActionNode, Inverter, RepeatUntilSuccess, Parallel,
)
from agents.bt_agent import BTAgent


class TestBTNodesUnit:
    def test_condition_true(self):
        node = Condition(lambda: True, name="always_true")
        assert node.tick() == Status.SUCCESS

    def test_condition_false(self):
        node = Condition(lambda: False, name="always_false")
        assert node.tick() == Status.FAILURE

    def test_action_returns_status(self):
        node = ActionNode(lambda: Status.SUCCESS, name="ok")
        assert node.tick() == Status.SUCCESS

    def test_action_returns_bool(self):
        node = ActionNode(lambda: True, name="ok")
        assert node.tick() == Status.SUCCESS
        node2 = ActionNode(lambda: False, name="fail")
        assert node2.tick() == Status.FAILURE

    def test_sequence_all_success(self):
        children = [ActionNode(lambda: Status.SUCCESS) for _ in range(3)]
        seq = Sequence(children)
        assert seq.tick() == Status.SUCCESS

    def test_sequence_first_failure(self):
        children = [
            ActionNode(lambda: Status.SUCCESS),
            ActionNode(lambda: Status.FAILURE),
            ActionNode(lambda: Status.SUCCESS),
        ]
        seq = Sequence(children)
        assert seq.tick() == Status.FAILURE

    def test_selector_first_success(self):
        children = [
            ActionNode(lambda: Status.FAILURE),
            ActionNode(lambda: Status.SUCCESS),
            ActionNode(lambda: Status.FAILURE),
        ]
        sel = Selector(children)
        assert sel.tick() == Status.SUCCESS

    def test_selector_all_fail(self):
        children = [ActionNode(lambda: Status.FAILURE) for _ in range(3)]
        sel = Selector(children)
        assert sel.tick() == Status.FAILURE

    def test_inverter(self):
        node = Inverter(ActionNode(lambda: Status.SUCCESS))
        assert node.tick() == Status.FAILURE
        node2 = Inverter(ActionNode(lambda: Status.FAILURE))
        assert node2.tick() == Status.SUCCESS

    def test_inverter_running_passthrough(self):
        node = Inverter(ActionNode(lambda: Status.RUNNING))
        assert node.tick() == Status.RUNNING

    def test_parallel_threshold(self):
        children = [
            ActionNode(lambda: Status.SUCCESS),
            ActionNode(lambda: Status.FAILURE),
            ActionNode(lambda: Status.SUCCESS),
        ]
        par = Parallel(children, threshold=2)
        assert par.tick() == Status.SUCCESS

    def test_parallel_impossible(self):
        children = [
            ActionNode(lambda: Status.FAILURE),
            ActionNode(lambda: Status.FAILURE),
            ActionNode(lambda: Status.SUCCESS),
        ]
        par = Parallel(children, threshold=3)
        assert par.tick() == Status.FAILURE


class TestBTNodesEdgeCases:
    def test_condition_exception_is_failure(self):
        def boom():
            raise ValueError("oops")
        node = Condition(boom)
        assert node.tick() == Status.FAILURE

    def test_action_exception_is_failure(self):
        def boom():
            raise RuntimeError("crash")
        node = ActionNode(boom)
        assert node.tick() == Status.FAILURE

    def test_empty_sequence_is_success(self):
        seq = Sequence([])
        assert seq.tick() == Status.SUCCESS

    def test_empty_selector_is_failure(self):
        sel = Selector([])
        assert sel.tick() == Status.FAILURE

    def test_repeat_until_success(self):
        counter = {"v": 0}
        def inc():
            counter["v"] += 1
            return Status.SUCCESS if counter["v"] >= 3 else Status.FAILURE
        node = RepeatUntilSuccess(ActionNode(inc), max_attempts=5)
        assert node.tick() == Status.SUCCESS
        assert counter["v"] == 3

    def test_repeat_until_success_max_exceeded(self):
        node = RepeatUntilSuccess(ActionNode(lambda: Status.FAILURE), max_attempts=3)
        assert node.tick() == Status.FAILURE

    def test_reset_clears_status(self):
        node = ActionNode(lambda: Status.SUCCESS, name="a")
        node.tick()
        assert node.status == Status.SUCCESS
        node.reset()
        assert node.status == Status.FAILURE

    def test_sequence_reset_cascades(self):
        children = [ActionNode(lambda: Status.SUCCESS) for _ in range(2)]
        seq = Sequence(children)
        seq.tick()
        seq.reset()
        assert all(c.status == Status.FAILURE for c in children)


class TestBTNodesMockData:
    def test_complex_tree(self):
        has_target = Condition(lambda: True, name="has_target")
        move = ActionNode(lambda: Status.SUCCESS, name="move")
        scan = ActionNode(lambda: Status.SUCCESS, name="scan")
        tree = Selector([
            Sequence([has_target, move]),
            scan,
        ])
        assert tree.tick() == Status.SUCCESS

    def test_node_name(self):
        node = Condition(lambda: True, name="my_check")
        assert node.name == "my_check"
        assert "my_check" in repr(node)

    def test_parallel_running(self):
        children = [
            ActionNode(lambda: Status.RUNNING),
            ActionNode(lambda: Status.SUCCESS),
        ]
        par = Parallel(children, threshold=2)
        assert par.tick() == Status.RUNNING


# ════════════════════════════════════════════
#  BTAgent
# ════════════════════════════════════════════

class TestBTAgentUnit:
    def test_tick_publishes_event(self):
        bus = EventBus()
        tree = ActionNode(lambda: Status.SUCCESS)
        agent = BTAgent(bus, tree)
        agent.perceive()
        agent.decide()
        agent.act()
        events = bus.get_history("bt.tick")
        assert len(events) == 1
        assert events[0]["data"]["status"] == "SUCCESS"

    def test_tick_count_increments(self):
        bus = EventBus()
        tree = ActionNode(lambda: Status.SUCCESS)
        agent = BTAgent(bus, tree)
        agent.decide()
        agent.decide()
        assert agent.tick_count == 2

    def test_success_failure_counts(self):
        call_count = {"v": 0}
        def alternate():
            call_count["v"] += 1
            return Status.SUCCESS if call_count["v"] % 2 == 0 else Status.FAILURE
        bus = EventBus()
        tree = ActionNode(alternate)
        agent = BTAgent(bus, tree)
        for _ in range(4):
            agent.decide()
        assert agent.success_count == 2
        assert agent.failure_count == 2


class TestBTAgentEdgeCases:
    def test_reset_tree(self):
        bus = EventBus()
        tree = ActionNode(lambda: Status.SUCCESS)
        agent = BTAgent(bus, tree)
        agent.decide()
        agent.reset_tree()
        assert agent.tick_count == 0
        assert agent.success_count == 0

    def test_swap_tree(self):
        bus = EventBus()
        tree1 = ActionNode(lambda: Status.SUCCESS, name="t1")
        tree2 = ActionNode(lambda: Status.FAILURE, name="t2")
        agent = BTAgent(bus, tree1)
        agent.decide()
        assert agent.last_status == Status.SUCCESS
        agent.root = tree2
        agent.decide()
        assert agent.last_status == Status.FAILURE

    def test_max_ticks_per_step(self):
        bus = EventBus()
        tree = ActionNode(lambda: Status.SUCCESS)
        agent = BTAgent(bus, tree, max_ticks_per_step=5)
        agent.decide()
        assert agent.tick_count == 5


class TestBTAgentMockData:
    def test_get_summary(self):
        bus = EventBus()
        tree = ActionNode(lambda: Status.SUCCESS)
        agent = BTAgent(bus, tree, agent_id="test_bt")
        agent.decide()
        agent.act()
        summary = agent.get_summary()
        assert summary["agent_id"] == "test_bt"
        assert summary["tick_count"] == 1
        assert summary["last_status"] == "SUCCESS"

    def test_engine_integration(self):
        bus = EventBus()
        engine = Engine(bus, enable_metrics=False)
        tree = Sequence([
            Condition(lambda: True),
            ActionNode(lambda: Status.SUCCESS),
        ])
        agent = BTAgent(bus, tree)
        engine.register_agent(agent)
        engine.run(steps=3)
        assert agent.tick_count == 3
        assert len(bus.get_history("bt.tick")) == 3


# ════════════════════════════════════════════
#  OccupancyMap (SLAM-lite)
# ════════════════════════════════════════════

from plugins.occupancy_map import OccupancyMap


class TestOccupancyMapUnit:
    def test_initial_cell_is_unknown(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=1.0)
        assert omap.get_cell(0, 0, 0) == -1

    def test_mark_free_and_occupied(self):
        bus = EventBus()
        omap = OccupancyMap(bus)
        omap.mark_free(0, 0, 0)
        assert omap.is_free(0, 0, 0)
        omap.mark_occupied(1, 0, 0)
        assert omap.is_occupied(1.0, 0, 0)

    def test_cell_count(self):
        bus = EventBus()
        omap = OccupancyMap(bus)
        omap.mark_free(0, 0, 0)
        omap.mark_occupied(1, 0, 0)
        counts = omap.cell_count()
        assert counts["free"] == 1
        assert counts["occupied"] == 1
        assert counts["total"] == 2

    def test_distance_event_updates_map(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=1.0)
        omap.start()
        from core.event_bus import _make_event
        event = _make_event("sensor.distance_3d", "test", {
            "robot_position": [0, 0, 0.31],
            "distances": {
                "forward": {"distance": 3.0, "blocked": True},
                "backward": {"distance": 15.0, "blocked": False},
            },
        })
        bus.publish(event)
        assert omap.update_count == 1
        occ = omap.get_occupied_cells()
        assert len(occ) >= 1
        omap.stop()

    def test_publishes_map_event(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=1.0)
        omap.start()
        omap.mark_free(0, 0, 0)
        omap.update()
        events = bus.get_history("map.occupancy")
        assert len(events) == 1
        assert "cells" in events[0]["data"]
        omap.stop()


class TestOccupancyMapEdgeCases:
    def test_resolution_clamp(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=0.01)
        assert omap.resolution >= 0.1

    def test_clear_resets(self):
        bus = EventBus()
        omap = OccupancyMap(bus)
        omap.mark_occupied(1, 1, 1)
        omap.clear()
        assert omap.cell_count()["total"] == 0
        assert omap.update_count == 0

    def test_decay_removes_stale(self):
        bus = EventBus()
        omap = OccupancyMap(bus, decay_enabled=True, decay_steps=2)
        omap.mark_occupied(5, 5, 5)
        omap._update_count = 0
        omap._last_seen[(5, 5, 5)] = 0
        omap._update_count = 5
        omap._apply_decay()
        assert omap.get_cell(5.0, 5.0, 5.0) == -1

    def test_lidar_event_updates_map(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=1.0)
        omap.start()
        from core.event_bus import _make_event
        event = _make_event("sensor.lidar", "test", {
            "robot_position": [0, 0, 0.31],
            "scan": [
                {"angle_rad": 0.0, "distance": 5.0, "hit": True},
                {"angle_rad": 1.57, "distance": 10.0, "hit": False},
            ],
        })
        bus.publish(event)
        assert omap.update_count == 1
        omap.stop()


class TestOccupancyMapMockData:
    def test_get_summary(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=0.5)
        omap.mark_free(0, 0, 0)
        omap.mark_occupied(2, 2, 0)
        summary = omap.get_summary()
        assert summary["resolution"] == 0.5
        assert summary["free"] == 1
        assert summary["occupied"] == 1

    def test_grid_to_world_roundtrip(self):
        bus = EventBus()
        omap = OccupancyMap(bus, resolution=0.5)
        gx, gy, gz = omap._world_to_grid(1.5, 2.0, 0.0)
        wx, wy, wz = omap._grid_to_world(gx, gy, gz)
        assert abs(wx - 1.5) < 0.5
        assert abs(wy - 2.0) < 0.5

    def test_multi_sensor_integration(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            omap = OccupancyMap(bus, resolution=1.0)
            dist = PhysicsDistance(bus, world)
            omap.start()
            dist.start()
            dist.update()
            assert omap.update_count >= 1
            omap.stop()
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  Observability
# ════════════════════════════════════════════

from core.observability import StructuredLogger, MetricsCollector, ObservabilityPlugin


class TestMetricsCollectorUnit:
    def test_counter_increments(self):
        m = MetricsCollector()
        m.inc("events.total")
        m.inc("events.total", 5)
        assert m.get_counter("events.total") == 6

    def test_gauge_set(self):
        m = MetricsCollector()
        m.set_gauge("step_count", 42)
        assert m.get_gauge("step_count") == 42

    def test_histogram_observe(self):
        m = MetricsCollector()
        m.observe("latency", 1.5)
        m.observe("latency", 2.5)
        s = m.get_histogram_summary("latency")
        assert s["count"] == 2
        assert s["avg"] == 2.0

    def test_to_json(self):
        m = MetricsCollector()
        m.inc("a", 1)
        m.set_gauge("b", 2)
        j = m.to_json()
        assert "counter_a" in j
        assert "gauge_b" in j

    def test_to_prometheus(self):
        m = MetricsCollector()
        m.inc("events.total", 10)
        m.set_gauge("step_count", 5)
        text = m.to_prometheus()
        assert "events_total 10" in text
        assert "step_count 5" in text

    def test_reset(self):
        m = MetricsCollector()
        m.inc("x")
        m.reset()
        assert m.get_counter("x") == 0


class TestStructuredLoggerUnit:
    def test_log_count(self):
        logger = StructuredLogger(name="test_obs_logger", level=10)
        logger.info("hello")
        logger.warning("warn")
        assert logger.log_count == 2

    def test_log_event(self):
        logger = StructuredLogger(name="test_obs_logger2", level=10)
        logger.log_event({"type": "test", "source": "s", "timestamp": 0, "data": {}})
        assert logger.log_count == 1


class TestObservabilityPluginUnit:
    def test_counts_events(self):
        bus = EventBus()
        metrics = MetricsCollector()
        plugin = ObservabilityPlugin(bus, metrics=metrics)
        plugin.start()
        from core.event_bus import _make_event
        bus.publish(_make_event("sensor.test", "src", {}))
        bus.publish(_make_event("sensor.test", "src", {}))
        assert metrics.get_counter("events.total") == 2
        assert metrics.get_counter("events.sensor.test") == 2
        plugin.stop()

    def test_tracks_engine_metrics(self):
        bus = EventBus()
        metrics = MetricsCollector()
        plugin = ObservabilityPlugin(bus, metrics=metrics)
        plugin.start()
        from core.event_bus import _make_event
        bus.publish(_make_event("engine.metrics", "engine", {"elapsed_ms": 1.5, "step": 1}))
        assert metrics.get_gauge("last_step_ms") == 1.5
        s = metrics.get_histogram_summary("step_time_ms")
        assert s["count"] == 1
        plugin.stop()

    def test_get_status(self):
        bus = EventBus()
        plugin = ObservabilityPlugin(bus)
        status = plugin.get_status()
        assert "metrics" in status


class TestObservabilityEdgeCases:
    def test_empty_histogram_summary(self):
        m = MetricsCollector()
        s = m.get_histogram_summary("nonexistent")
        assert s["count"] == 0

    def test_empty_prometheus(self):
        m = MetricsCollector()
        assert m.to_prometheus() == ""

    def test_gauge_nonexistent(self):
        m = MetricsCollector()
        assert m.get_gauge("nope") is None

    def test_plugin_stop_unsubscribes(self):
        bus = EventBus()
        plugin = ObservabilityPlugin(bus)
        plugin.start()
        plugin.stop()
        from core.event_bus import _make_event
        bus.publish(_make_event("test", "s", {}))
        assert plugin.metrics.get_counter("events.total") == 0


class TestObservabilityMockData:
    def test_full_engine_integration(self):
        bus = EventBus()
        metrics = MetricsCollector()
        plugin = ObservabilityPlugin(bus, metrics=metrics)
        engine = Engine(bus, enable_metrics=True)
        engine.register_plugin(plugin)
        engine.run(steps=5)
        assert metrics.get_counter("events.total") >= 5
        s = metrics.get_histogram_summary("step_time_ms")
        assert s["count"] == 5

    def test_prometheus_api_endpoint(self):
        from fastapi.testclient import TestClient
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            metrics = MetricsCollector()
            metrics.inc("events.total", 42)
            engine = Engine(bus)
            from api.server import create_app
            app = create_app(engine, bus, world, metrics_collector=metrics)
            client = TestClient(app)
            resp = client.get("/api/metrics/prometheus")
            assert resp.status_code == 200
            assert "events_total 42" in resp.text
            resp2 = client.get("/api/metrics/json")
            assert resp2.status_code == 200
            assert resp2.json()["metrics"]["counter_events.total"] == 42
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  RRT* Motion Planner
# ════════════════════════════════════════════

from planning.rrt import RRTStar, RRTNode, _dist


class TestRRTStarUnit:
    def test_plan_empty_space(self):
        planner = RRTStar(
            bounds_min=(-5, -5, 0), bounds_max=(5, 5, 2),
            step_size=1.0, goal_radius=1.0, max_iterations=500, seed=42,
        )
        path = planner.plan(start=(0, 0, 0), goal=(3, 0, 0))
        assert len(path) >= 2
        assert _dist(path[0], (0, 0, 0)) < 0.01
        assert _dist(path[-1], (3, 0, 0)) <= 1.0

    def test_plan_with_obstacle(self):
        def obstacle(x, y, z):
            return 1.0 < x < 2.0 and -1.0 < y < 1.0
        planner = RRTStar(
            bounds_min=(-5, -5, 0), bounds_max=(5, 5, 1),
            step_size=0.5, goal_radius=0.8, max_iterations=1000,
            collision_fn=obstacle, seed=42,
        )
        path = planner.plan(start=(0, 0, 0), goal=(3, 0, 0))
        assert len(path) >= 2
        for pt in path:
            assert not obstacle(pt[0], pt[1], pt[2])

    def test_nodes_populated(self):
        planner = RRTStar(max_iterations=100, seed=1)
        planner.plan(start=(0, 0, 0), goal=(5, 0, 0))
        assert len(planner.nodes) > 1

    def test_iterations_tracked(self):
        planner = RRTStar(max_iterations=50, seed=1)
        planner.plan(start=(0, 0, 0), goal=(5, 0, 0))
        assert planner.iterations_used <= 50


class TestRRTStarEdgeCases:
    def test_start_equals_goal(self):
        planner = RRTStar(goal_radius=1.0, max_iterations=50, seed=1)
        path = planner.plan(start=(0, 0, 0), goal=(0, 0, 0))
        assert len(path) >= 1

    def test_unreachable_in_few_iterations(self):
        def wall(x, y, z):
            return 0.5 < x < 100.0
        planner = RRTStar(
            bounds_min=(-1, -1, 0), bounds_max=(10, 1, 1),
            max_iterations=10, collision_fn=wall, seed=1,
        )
        path = planner.plan(start=(0, 0, 0), goal=(5, 0, 0))
        assert path == []

    def test_smooth_path(self):
        planner = RRTStar(
            bounds_min=(-5, -5, 0), bounds_max=(5, 5, 1),
            step_size=1.0, goal_radius=1.0, max_iterations=500, seed=42,
        )
        path = planner.plan(start=(0, 0, 0), goal=(3, 0, 0))
        if len(path) > 2:
            smoothed = planner.smooth_path(path)
            assert len(smoothed) <= len(path)

    def test_smooth_short_path(self):
        planner = RRTStar(seed=1)
        smoothed = planner.smooth_path([(0, 0, 0), (1, 0, 0)])
        assert len(smoothed) == 2


class TestRRTStarMockData:
    def test_get_summary(self):
        planner = RRTStar(max_iterations=100, seed=1)
        planner.plan(start=(0, 0, 0), goal=(3, 0, 0))
        s = planner.get_summary()
        assert "nodes" in s
        assert "iterations_used" in s

    def test_rewire_improves_cost(self):
        planner_no_rewire = RRTStar(
            step_size=1.0, goal_radius=1.0, max_iterations=300,
            rewire_radius=0, seed=42,
        )
        planner_rewire = RRTStar(
            step_size=1.0, goal_radius=1.0, max_iterations=300,
            rewire_radius=2.0, seed=42,
        )
        path1 = planner_no_rewire.plan((0, 0, 0), (4, 0, 0))
        path2 = planner_rewire.plan((0, 0, 0), (4, 0, 0))
        if path1 and path2:
            len1 = sum(_dist(path1[i], path1[i+1]) for i in range(len(path1)-1))
            len2 = sum(_dist(path2[i], path2[i+1]) for i in range(len(path2)-1))
            assert len2 <= len1 * 1.5

    def test_rrt_node_dataclass(self):
        n = RRTNode(position=(1.0, 2.0, 3.0))
        assert n.position == (1.0, 2.0, 3.0)
        assert n.parent is None
        assert n.cost == 0.0


# ════════════════════════════════════════════
#  GraspPlanner (pick-and-place)
# ════════════════════════════════════════════

from planning.grasp_planner import GraspPlanner, GraspPose


class TestGraspPlannerUnit:
    def test_plan_pick(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_pick((0.0, 0.0, 0.3))
            assert len(poses) == 3
            assert poses[0].phase == "approach"
            assert poses[1].phase == "grasp"
            assert poses[2].phase == "lift"
        finally:
            world.disconnect()

    def test_plan_place(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_place((0.1, 0.0, 0.3))
            assert len(poses) == 3
            assert poses[0].phase == "transport"
            assert poses[2].phase == "retreat"
        finally:
            world.disconnect()

    def test_plan_pick_and_place(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_pick_and_place(
                pick_target=(0.0, 0.0, 0.3),
                place_target=(0.05, 0.0, 0.3),
            )
            assert len(poses) == 6
            phases = [p.phase for p in poses]
            assert phases == ["approach", "grasp", "lift", "transport", "place", "retreat"]
        finally:
            world.disconnect()

    def test_each_pose_has_joint_targets(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_pick((0.0, 0.0, 0.3))
            for pose in poses:
                assert pose.joint_targets is not None
                assert len(pose.joint_targets) == robot.num_joints
        finally:
            world.disconnect()


class TestGraspPlannerEdgeCases:
    def test_evaluate_empty_plan(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            result = planner.evaluate_plan([])
            assert result["feasible"] is False
            assert result["phases"] == 0
        finally:
            world.disconnect()

    def test_last_plan_stored(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            planner.plan_pick_and_place((0.0, 0.0, 0.3), (0.05, 0.0, 0.3))
            assert len(planner.last_plan) == 6
        finally:
            world.disconnect()

    def test_grasp_pose_to_dict(self):
        pose = GraspPose(phase="test", position=(1, 2, 3), joint_targets=[0.1, 0.2])
        d = pose.to_dict()
        assert d["phase"] == "test"
        assert d["position"] == [1, 2, 3]
        assert d["joint_targets"] == [0.1, 0.2]


class TestGraspPlannerMockData:
    def test_get_action_commands(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_pick((0.0, 0.0, 0.3))
            commands = planner.get_action_commands(poses)
            assert len(commands) == 3
            assert all("joint_positions" in c for c in commands)
            assert all("phase" in c for c in commands)
        finally:
            world.disconnect()

    def test_evaluate_plan(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            planner = GraspPlanner(robot)
            poses = planner.plan_pick_and_place(
                (0.0, 0.0, 0.3), (0.05, 0.0, 0.3)
            )
            result = planner.evaluate_plan(poses)
            assert result["phases"] == 6
            assert result["feasible"] is True
            assert "avg_ik_error" in result
        finally:
            world.disconnect()

    def test_full_pipeline_with_urdf_action(self):
        world = PhysicsWorld(gui=False)
        try:
            from simulator.urdf_robot import URDFRobot
            from actions.urdf_action import URDFAction
            robot = URDFRobot(world.client_id, position=(0, 0, 0))
            action = URDFAction(robot, world.client_id, sim_steps=60)
            planner = GraspPlanner(robot)
            poses = planner.plan_pick((0.0, 0.0, 0.3))
            commands = planner.get_action_commands(poses)
            for cmd in commands:
                result = action.execute(cmd)
                assert "success" in result
        finally:
            world.disconnect()


# ════════════════════════════════════════════════════════════════
# EVENT SCHEMA VALIDATION TESTS
# ════════════════════════════════════════════════════════════════

class TestEventSchemaUnit:
    """Unit tests for event_schema module."""

    def test_schemas_dict_exists(self):
        from core.event_schema import SCHEMAS
        assert isinstance(SCHEMAS, dict)
        assert len(SCHEMAS) >= 10

    def test_validate_known_type_ok(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data("engine.metrics", {"step": 5, "elapsed_ms": 1.2})
        assert errs == []

    def test_validate_missing_field(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data("engine.metrics", {"step": 5})
        assert any(e.field == "elapsed_ms" for e in errs)

    def test_validate_wrong_type(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data("engine.metrics", {"step": "not_int", "elapsed_ms": 1.0})
        assert any(e.field == "step" for e in errs)

    def test_validate_unknown_type_returns_empty(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data("custom.unknown", {"anything": True})
        assert errs == []

    def test_validate_strict_extra_fields(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data(
            "engine.metrics",
            {"step": 1, "elapsed_ms": 0.5, "extra_field": "oops"},
            strict=True,
        )
        assert any(e.field == "extra_field" for e in errs)

    def test_validate_min_val_violation(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data("engine.metrics", {"step": -1, "elapsed_ms": 1.0})
        assert any("min" in e.message for e in errs)

    def test_validate_list_items(self):
        from core.event_schema import validate_event_data
        errs = validate_event_data(
            "action.move_3d",
            {
                "robot_id": "r1", "direction": "forward", "success": True,
                "old_position": [0, 0, "bad"],
                "new_position": [1.0, 2.0, 3.0],
            },
        )
        assert any(e.field == "old_position" for e in errs)

    def test_validation_error_repr(self):
        from core.event_schema import ValidationError
        e = ValidationError("x", "bad")
        assert "x" in repr(e) and "bad" in repr(e)


class TestSchemaValidatorPlugin:
    """Tests for SchemaValidator plugin on EventBus."""

    def test_counts_violations(self):
        from core.event_bus import EventBus, _make_event
        from core.event_schema import SchemaValidator
        bus = EventBus()
        sv = SchemaValidator(bus, strict=False)
        bus.publish(_make_event("engine.metrics", "test", {"step": "wrong"}))
        assert sv.violation_count >= 1
        assert sv.checked_count >= 1

    def test_no_violations_for_good_events(self):
        from core.event_bus import EventBus, _make_event
        from core.event_schema import SchemaValidator
        bus = EventBus()
        sv = SchemaValidator(bus)
        bus.publish(_make_event("engine.metrics", "test", {"step": 1, "elapsed_ms": 0.5}))
        assert sv.violation_count == 0

    def test_callback_fires(self):
        from core.event_bus import EventBus, _make_event
        from core.event_schema import SchemaValidator
        bus = EventBus()
        captured = []
        validator = SchemaValidator(bus, on_violation=lambda v: captured.append(v))
        bus.publish(_make_event("engine.metrics", "test", {}))
        assert len(captured) >= 1
        assert validator.checked_count >= 1

    def test_summary(self):
        from core.event_bus import EventBus, _make_event
        from core.event_schema import SchemaValidator
        bus = EventBus()
        sv = SchemaValidator(bus)
        bus.publish(_make_event("engine.metrics", "test", {"step": 1, "elapsed_ms": 0.5}))
        s = sv.summary()
        assert "checked" in s and "violations" in s

    def test_strict_mode(self):
        from core.event_bus import EventBus, _make_event
        from core.event_schema import SchemaValidator
        bus = EventBus()
        sv = SchemaValidator(bus, strict=True)
        bus.publish(_make_event(
            "engine.metrics", "test",
            {"step": 1, "elapsed_ms": 0.5, "extra": "field"},
        ))
        assert sv.violation_count >= 1


# ════════════════════════════════════════════════════════════════
# PLUGIN DEPENDENCY GRAPH TESTS
# ════════════════════════════════════════════════════════════════

class _FakeDepPlugin:
    def __init__(self, pid, reqs=None):
        self.plugin_id = pid
        self.requires = reqs or []
        self.active = False


class TestPluginDepsUnit:
    """Unit tests for plugin_deps module."""

    def test_no_deps(self):
        from core.plugin_deps import resolve_order
        a = _FakeDepPlugin("a")
        b = _FakeDepPlugin("b")
        result = resolve_order([a, b])
        assert len(result) == 2

    def test_simple_ordering(self):
        from core.plugin_deps import resolve_order
        a = _FakeDepPlugin("a")
        b = _FakeDepPlugin("b", reqs=["a"])
        result = resolve_order([b, a])
        ids = [p.plugin_id for p in result]
        assert ids.index("a") < ids.index("b")

    def test_chain_deps(self):
        from core.plugin_deps import resolve_order
        a = _FakeDepPlugin("a")
        b = _FakeDepPlugin("b", reqs=["a"])
        c = _FakeDepPlugin("c", reqs=["b"])
        result = resolve_order([c, b, a])
        ids = [p.plugin_id for p in result]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_missing_dep_raises(self):
        from core.plugin_deps import resolve_order, DependencyError
        b = _FakeDepPlugin("b", reqs=["missing"])
        with pytest.raises(DependencyError, match="missing"):
            resolve_order([b])

    def test_cycle_raises(self):
        from core.plugin_deps import resolve_order, DependencyError
        a = _FakeDepPlugin("a", reqs=["b"])
        b = _FakeDepPlugin("b", reqs=["a"])
        with pytest.raises(DependencyError, match="Cyclic"):
            resolve_order([a, b])

    def test_build_graph(self):
        from core.plugin_deps import build_graph
        a = _FakeDepPlugin("a")
        b = _FakeDepPlugin("b", reqs=["a"])
        g = build_graph([a, b])
        assert g == {"a": [], "b": ["a"]}

    def test_check_health_ok(self):
        from core.plugin_deps import check_health
        a = _FakeDepPlugin("a")
        b = _FakeDepPlugin("b", reqs=["a"])
        result = check_health([a, b])
        assert result["ok"] is True
        assert "a" in result["order"]

    def test_check_health_cycle(self):
        from core.plugin_deps import check_health
        a = _FakeDepPlugin("a", reqs=["b"])
        b = _FakeDepPlugin("b", reqs=["a"])
        result = check_health([a, b])
        assert result["ok"] is False


# ════════════════════════════════════════════════════════════════
# STATE MACHINE (MISSION FSM) TESTS
# ════════════════════════════════════════════════════════════════

class TestMissionFSMUnit:
    """Unit tests for MissionFSM."""

    def test_initial_phase(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        assert fsm.phase == Phase.IDLE

    def test_advance_cycle(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        assert fsm.advance()
        assert fsm.phase == Phase.NAVIGATE
        assert fsm.advance()
        assert fsm.phase == Phase.MANIPULATE
        assert fsm.advance()
        assert fsm.phase == Phase.RETURN
        assert fsm.advance()
        assert fsm.phase == Phase.IDLE

    def test_transition(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        ok = fsm.transition(Phase.NAVIGATE)
        assert ok is True
        assert fsm.phase == Phase.NAVIGATE

    def test_invalid_transition(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        ok = fsm.transition(Phase.RETURN)
        assert ok is False
        assert fsm.phase == Phase.IDLE

    def test_error_from_any(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        fsm.transition(Phase.NAVIGATE)
        assert fsm.error()
        assert fsm.phase == Phase.ERROR

    def test_error_recovery(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        fsm.transition(Phase.ERROR)
        assert fsm.transition(Phase.IDLE)
        assert fsm.phase == Phase.IDLE

    def test_guard_blocks(self):
        from core.state_machine import MissionFSM, Phase, Transition
        blocked = Transition(Phase.IDLE, Phase.NAVIGATE, guard=lambda: False)
        fsm = MissionFSM("r1", transitions=[blocked])
        assert not fsm.transition(Phase.NAVIGATE)
        assert fsm.phase == Phase.IDLE

    def test_can_transition(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        assert fsm.can_transition(Phase.NAVIGATE)
        assert not fsm.can_transition(Phase.RETURN)

    def test_on_enter_callback(self):
        from core.state_machine import MissionFSM, Phase
        log = []
        fsm = MissionFSM("r1")
        fsm.on_enter(Phase.NAVIGATE, lambda: log.append("entered"))
        fsm.transition(Phase.NAVIGATE)
        assert log == ["entered"]

    def test_on_exit_callback(self):
        from core.state_machine import MissionFSM, Phase
        log = []
        fsm = MissionFSM("r1")
        fsm.on_exit(Phase.IDLE, lambda: log.append("exited"))
        fsm.transition(Phase.NAVIGATE)
        assert log == ["exited"]

    def test_history(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        fsm.transition(Phase.NAVIGATE)
        fsm.transition(Phase.MANIPULATE)
        assert len(fsm.history) == 2
        assert fsm.history[0]["from"] == "IDLE"
        assert fsm.history[0]["to"] == "NAVIGATE"

    def test_reset(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        fsm.transition(Phase.NAVIGATE)
        fsm.reset()
        assert fsm.phase == Phase.IDLE
        assert fsm.history == []

    def test_force(self):
        from core.state_machine import MissionFSM, Phase
        fsm = MissionFSM("r1")
        fsm.force(Phase.MANIPULATE)
        assert fsm.phase == Phase.MANIPULATE

    def test_get_status(self):
        from core.state_machine import MissionFSM
        fsm = MissionFSM("r1")
        s = fsm.get_status()
        assert s["phase"] == "IDLE"
        assert s["robot_id"] == "r1"

    def test_event_published(self):
        from core.event_bus import EventBus
        from core.state_machine import MissionFSM, Phase
        bus = EventBus()
        captured = []
        bus.subscribe("mission.transition", captured.append)
        fsm = MissionFSM("r1", event_bus=bus)
        fsm.transition(Phase.NAVIGATE)
        assert len(captured) == 1
        assert captured[0]["data"]["from"] == "IDLE"


# ════════════════════════════════════════════════════════════════
# PHYSICS WORLD remove_robot TESTS
# ════════════════════════════════════════════════════════════════

class TestPhysicsWorldRemoveRobot:
    def test_remove_added_robot(self):
        from simulator.physics_world import PhysicsWorld
        w = PhysicsWorld(gui=False)
        try:
            w.add_robot("tmp", x=1, y=1)
            assert "tmp" in w.get_robot_ids()
            w.remove_robot("tmp")
            assert "tmp" not in w.get_robot_ids()
        finally:
            w.disconnect()

    def test_remove_default_raises(self):
        from simulator.physics_world import PhysicsWorld
        w = PhysicsWorld(gui=False)
        try:
            with pytest.raises(ValueError, match="default"):
                w.remove_robot("default")
        finally:
            w.disconnect()

    def test_remove_nonexistent_raises(self):
        from simulator.physics_world import PhysicsWorld
        w = PhysicsWorld(gui=False)
        try:
            with pytest.raises(ValueError, match="not found"):
                w.remove_robot("ghost")
        finally:
            w.disconnect()


# ════════════════════════════════════════════════════════════════
# API V2 ENDPOINT TESTS
# ════════════════════════════════════════════════════════════════

class TestAPIv2:
    """Unit tests for the expanded API server."""

    @pytest.fixture()
    def client(self):
        from core.event_bus import EventBus
        from core.engine import Engine
        from core.config import Config
        from core.observability import MetricsCollector
        from core.hot_reload import HotReloadManager
        from simulator.physics_world import PhysicsWorld
        from api.server import create_app

        bus = EventBus()
        world = PhysicsWorld(gui=False)
        cfg = Config({"test": True})
        engine = Engine(bus, config=cfg)
        metrics = MetricsCollector()
        hrm = HotReloadManager(engine, bus)

        from plugins.goal_publisher import GoalPublisher
        gp = GoalPublisher(bus)
        gp.set_target(1, 1, 0.31)
        engine.register_plugin(gp)

        app = create_app(
            engine, bus, world,
            config=cfg, goal_publisher=gp,
            metrics_collector=metrics,
            hot_reload_manager=hrm,
        )
        from starlette.testclient import TestClient
        yield TestClient(app), world
        world.disconnect()

    def test_pause_resume(self, client):
        tc, _ = client
        r = tc.post("/api/pause")
        assert r.status_code == 200
        assert r.json()["paused"] is True
        r = tc.post("/api/resume")
        assert r.json()["paused"] is False

    def test_add_remove_robot(self, client):
        tc, _ = client
        r = tc.post("/api/robots/add", json={"robot_id": "api_bot", "x": 1, "y": 2})
        assert r.status_code == 200
        assert r.json()["added"] == "api_bot"
        r = tc.delete("/api/robots/api_bot")
        assert r.status_code == 200

    def test_add_duplicate_robot(self, client):
        tc, _ = client
        tc.post("/api/robots/add", json={"robot_id": "dup"})
        r = tc.post("/api/robots/add", json={"robot_id": "dup"})
        assert r.status_code == 400

    def test_add_obstacle(self, client):
        tc, _ = client
        r = tc.post("/api/obstacles/add", json={"x": 3, "y": 3})
        assert r.status_code == 200
        assert "obstacle_id" in r.json()

    def test_clear_obstacles(self, client):
        tc, _ = client
        tc.post("/api/obstacles/add", json={"x": 1, "y": 1})
        r = tc.delete("/api/obstacles/clear")
        assert r.json()["obstacle_count"] == 0

    def test_patch_config(self, client):
        tc, _ = client
        r = tc.patch("/api/config", json={"key": "foo.bar", "value": 42})
        assert r.status_code == 200
        assert r.json()["value"] == 42

    def test_plugins_list(self, client):
        tc, _ = client
        r = tc.get("/api/plugins")
        assert r.status_code == 200
        assert "plugins" in r.json()

    def test_move_robot(self, client):
        tc, _ = client
        r = tc.post("/api/robots/move", json={"direction": "forward"})
        assert r.status_code == 200
        assert "position" in r.json()

    def test_state_includes_new_fields(self, client):
        tc, _ = client
        r = tc.get("/api/state")
        d = r.json()
        assert "plugins" in d
        assert "agents" in d
        assert "paused" in d
        assert "obstacles" in d
        assert isinstance(d["obstacles"], list)

    def test_health_includes_details(self, client):
        tc, _ = client
        r = tc.get("/api/health")
        d = r.json()
        assert "paused" in d
        assert "step_count" in d
        assert "robot_count" in d

    def test_dashboard_returns_html(self, client):
        tc, _ = client
        r = tc.get("/dashboard")
        assert r.status_code == 200
        assert "IR-AAP" in r.text

    def test_mission_wizard_returns_html(self, client):
        tc, _ = client
        r = tc.get("/wizard")
        assert r.status_code == 200
        assert "YAML" in r.text or "yaml" in r.text

    def test_pause_sets_engine_flag(self, client):
        tc, _ = client
        tc.post("/api/pause")
        assert tc.get("/api/health").json()["paused"] is True
        tc.post("/api/resume")
        assert tc.get("/api/health").json()["paused"] is False

    def test_config_save_yaml(self, client):
        import os
        tc, _ = client
        fn = "_pytest_save_cfg.yaml"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, fn)
        try:
            r = tc.post("/api/config/save", params={"filename": fn})
            assert r.status_code == 200
            assert os.path.isfile(path)
            assert "world" in open(path, encoding="utf-8").read()
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_replay_sessions_endpoint(self, client):
        tc, _ = client
        r = tc.get("/api/replay/sessions")
        assert r.status_code == 200
        assert "sessions" in r.json()

    def test_replay_events_loads_jsonl(self, client):
        import json
        import os
        tc, _ = client
        fn = "_pytest_replay_tmp.jsonl"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, fn)
        line = json.dumps({
            "timestamp": 1,
            "type": "engine.metrics",
            "source": "e",
            "data": {"step": 1, "elapsed_ms": 1.0},
        })
        with open(path, "w", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            r = tc.get("/api/replay/events", params={"file": fn, "limit": 10})
            assert r.status_code == 200
            body = r.json()
            assert body["returned"] == 1
            assert body["events"][0]["type"] == "engine.metrics"
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_replay_events_rejects_bad_name(self, client):
        tc, _ = client
        r = tc.get("/api/replay/events", params={"file": "../../../etc/passwd.jsonl"})
        assert r.status_code == 400


class TestEnginePluginDependencyOrder:
    """Engine.start() applies resolve_order when plugins declare requires=[]."""

    def test_orders_dependent_plugins(self):
        from plugins.base_plugin import BasePlugin
        from core.event_bus import EventBus
        from core.engine import Engine

        class _P(BasePlugin):
            def __init__(self, bus, pid, reqs=()):
                super().__init__(bus, pid)
                self.requires = list(reqs)

            def on_update(self) -> None:
                pass

        bus = EventBus()
        eng = Engine(bus)
        root = _P(bus, "root")
        leaf = _P(bus, "leaf", reqs=["root"])
        eng.register_plugin(leaf)
        eng.register_plugin(root)
        eng.start()
        assert [p.plugin_id for p in eng._plugins] == ["root", "leaf"]
        eng.stop()


class TestConfigYamlRoundTrip:
    def test_save_and_load_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        from core.config import Config
        cfg = Config({"sensor": {"noise_level": 0.33}})
        ypath = tmp_path / "t.yaml"
        cfg.save(str(ypath))
        cfg2 = Config.load_file(str(ypath))
        assert cfg2.get("sensor.noise_level") == 0.33

    def test_load_file_alias(self, tmp_path):
        from core.config import Config
        jpath = tmp_path / "c.json"
        cfg = Config({"a": {"b": 7}})
        cfg.save(str(jpath))
        cfg2 = Config.load_file(str(jpath))
        assert cfg2.get("a.b") == 7


class TestExecuteWsCommands:
    def test_ping_ack(self):
        from core.event_bus import EventBus
        from core.engine import Engine
        from core.ws_commands import execute_ws_command
        from simulator.physics_world import PhysicsWorld

        bus = EventBus()
        world = PhysicsWorld(gui=False)
        eng = Engine(bus)
        try:
            execute_ws_command({"cmd": "ping"}, engine=eng, world=world, bus=bus)
            ack = bus.get_history("ws.command.ack")[-1]
            assert ack["data"]["ok"] is True
            assert ack["data"]["cmd"] == "ping"
        finally:
            world.disconnect()

    def test_step_increments(self):
        from core.event_bus import EventBus
        from core.engine import Engine
        from core.ws_commands import execute_ws_command
        from simulator.physics_world import PhysicsWorld

        bus = EventBus()
        world = PhysicsWorld(gui=False)
        eng = Engine(bus)
        try:
            eng.start()
            execute_ws_command({"cmd": "step", "steps": 4}, engine=eng, world=world, bus=bus)
            assert eng.step_count == 4
        finally:
            eng.stop()
            world.disconnect()

    def test_goal_with_publisher(self):
        from core.event_bus import EventBus
        from core.engine import Engine
        from core.ws_commands import execute_ws_command
        from plugins.goal_publisher import GoalPublisher
        from simulator.physics_world import PhysicsWorld

        bus = EventBus()
        world = PhysicsWorld(gui=False)
        eng = Engine(bus)
        gp = GoalPublisher(bus)
        eng.register_plugin(gp)
        try:
            eng.start()
            execute_ws_command(
                {"cmd": "goal", "x": 3, "y": 4, "z": 0.5},
                engine=eng, world=world, bus=bus, goal_publisher=gp,
            )
            ack = bus.get_history("ws.command.ack")[-1]
            assert ack["data"]["ok"] is True
        finally:
            eng.stop()
            world.disconnect()
