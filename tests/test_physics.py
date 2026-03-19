"""
test_physics.py — full test suite for the 3D physics modules.

Per 測試規則.txt, every module includes:
  1. Unit tests
  2. Edge case tests
  3. Mock data tests

Run:  pytest tests/test_physics.py -v
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, _make_event
from core.engine import Engine

# World interface
from simulator.world_interface import WorldInterface

# Physics modules
from simulator.physics_world import PhysicsWorld, DIRECTIONS_3D, _RAY_MAX_DISTANCE
from plugins.physics_camera import PhysicsCamera
from plugins.physics_distance import PhysicsDistance
from agents.physics_agent import PhysicsAgent, PREFERRED_DIRECTIONS_3D


# ════════════════════════════════════════════
#  WorldInterface ABC
# ════════════════════════════════════════════

class TestWorldInterfaceUnit:
    def test_gridworld_is_world_interface(self):
        from simulator.world import GridWorld
        world = GridWorld(5, 5)
        assert isinstance(world, WorldInterface)

    def test_physics_world_is_world_interface(self):
        world = PhysicsWorld(gui=False)
        try:
            assert isinstance(world, WorldInterface)
        finally:
            world.disconnect()

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            WorldInterface()


class TestWorldInterfaceEdgeCases:
    def test_gridworld_step_physics_noop(self):
        from simulator.world import GridWorld
        world = GridWorld(3, 3)
        world.step_physics()
        assert world.get_robot_position() == (0, 0)


class TestWorldInterfaceMockData:
    def test_both_worlds_have_render(self):
        from simulator.world import GridWorld
        gw = GridWorld(3, 3)
        assert isinstance(gw.render(), str)

        pw = PhysicsWorld(gui=False)
        try:
            assert isinstance(pw.render(), str)
        finally:
            pw.disconnect()


# ════════════════════════════════════════════
#  PhysicsWorld
# ════════════════════════════════════════════

class TestPhysicsWorldUnit:
    def test_create_headless(self):
        world = PhysicsWorld(gui=False)
        try:
            pos = world.get_robot_position()
            assert len(pos) == 3
            assert pos[2] > 0
        finally:
            world.disconnect()

    def test_place_obstacle(self):
        world = PhysicsWorld(gui=False)
        try:
            oid = world.place_obstacle(5, 0, 0.5)
            assert oid >= 0
            assert len(world._obstacle_ids) == 1
        finally:
            world.disconnect()

    def test_clear_obstacles(self):
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(5, 0, 0.5)
            world.place_obstacle(3, 3, 0.5)
            assert len(world._obstacle_ids) == 2
            world.clear_obstacles()
            assert len(world._obstacle_ids) == 0
        finally:
            world.disconnect()

    def test_get_obstacles_for_visualization(self):
        world = PhysicsWorld(gui=False)
        try:
            assert world.get_obstacles() == []
            world.place_obstacle(1.0, 2.0, 0.5)
            obs = world.get_obstacles()
            assert len(obs) == 1
            assert "position" in obs[0] and "half_extent" in obs[0]
            assert abs(obs[0]["position"][0] - 1.0) < 0.05
            assert abs(obs[0]["position"][1] - 2.0) < 0.05
            assert obs[0]["half_extent"] == 0.5
        finally:
            world.disconnect()

    def test_move_robot_changes_position(self):
        world = PhysicsWorld(gui=False)
        try:
            pos_before = world.get_robot_position()
            world.move_robot("forward")
            pos_after = world.get_robot_position()
            assert math.dist(pos_before, pos_after) > 0.01
        finally:
            world.disconnect()

    def test_step_physics(self):
        world = PhysicsWorld(gui=False)
        try:
            world.step_physics()
        finally:
            world.disconnect()

    def test_render_returns_string(self):
        world = PhysicsWorld(gui=False)
        try:
            r = world.render()
            assert "PhysicsWorld" in r
            assert "default" in r
        finally:
            world.disconnect()

    def test_reset(self):
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(5, 0, 0.5)
            world.move_robot("forward")
            world.reset()
            assert len(world._obstacle_ids) == 0
            pos = world.get_robot_position()
            assert abs(pos[0]) < 0.1
            assert abs(pos[1]) < 0.1
        finally:
            world.disconnect()

    def test_set_robot_position(self):
        world = PhysicsWorld(gui=False)
        try:
            world.set_robot_position(3.0, 4.0)
            pos = world.get_robot_position()
            assert abs(pos[0] - 3.0) < 0.1
            assert abs(pos[1] - 4.0) < 0.1
        finally:
            world.disconnect()

    def test_get_surroundings_returns_all_directions(self):
        world = PhysicsWorld(gui=False)
        try:
            s = world.get_surroundings(0, 0, 0)
            for d in DIRECTIONS_3D:
                assert d in s
                assert "distance" in s[d]
                assert "blocked" in s[d]
        finally:
            world.disconnect()


class TestPhysicsWorldDynamics:
    def test_spawn_dynamic_box_tracked(self):
        world = PhysicsWorld(gui=False)
        try:
            world.spawn_dynamic_box(1, 2, 0.5, half_extent=0.08, mass=0.2)
            objs = world.get_dynamic_objects()
            assert len(objs) == 1
            assert objs[0]["position"][2] > 0
            assert "linear_velocity" in objs[0]
        finally:
            world.disconnect()

    def test_dynamic_body_falls_with_gravity(self):
        world = PhysicsWorld(gui=False, timestep=1 / 240, gravity=-9.81)
        try:
            world.spawn_dynamic_box(0, 0, 2.5, half_extent=0.05, mass=0.15)
            z0 = world.get_dynamic_objects()[0]["position"][2]
            for _ in range(360):
                world.step_physics()
            z1 = world.get_dynamic_objects()[0]["position"][2]
            assert z1 < z0 - 0.2
        finally:
            world.disconnect()

    def test_reset_clears_dynamic_objects(self):
        world = PhysicsWorld(gui=False)
        try:
            world.spawn_dynamic_box(0, 0, 1.0, mass=0.1, half_extent=0.05)
            world.reset()
            assert world.get_dynamic_objects() == []
        finally:
            world.disconnect()

    def test_profile_yaml_dynamics_block(self):
        from core.profile_loader import build_world_from_profile

        spec = {
            "backend": "pybullet",
            "gui": False,
            "physics": {"timestep": 0.004167, "gravity_z": -9.81},
            "dynamics": [{"x": 0.1, "y": -0.2, "z": 1.2, "mass": 0.1, "half_extent": 0.04}],
        }
        world = build_world_from_profile(spec)
        try:
            assert len(world.get_dynamic_objects()) == 1
        finally:
            world.disconnect()


class TestPhysicsWorldEdgeCases:
    def test_invalid_direction(self):
        world = PhysicsWorld(gui=False)
        try:
            with pytest.raises(ValueError, match="Unknown direction"):
                world.move_robot("diagonal")
        finally:
            world.disconnect()

    def test_disconnect_twice_safe(self):
        world = PhysicsWorld(gui=False)
        world.disconnect()
        world.disconnect()

    def test_multiple_obstacles_same_location(self):
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 2, 0.5)
            world.place_obstacle(2, 2, 1.5)
            assert len(world._obstacle_ids) == 2
        finally:
            world.disconnect()

    def test_is_obstacle_empty_world(self):
        world = PhysicsWorld(gui=False)
        try:
            assert not world.is_obstacle(5, 5, 0.5)
        finally:
            world.disconnect()


class TestPhysicsWorldMockData:
    def test_obstacle_course(self):
        """Build a small obstacle course and verify surroundings detect blocks."""
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 0, 0.5)
            s = world.get_surroundings(0, 0, 0)
            forward_dist = s["forward"]["distance"]
            assert forward_dist < _RAY_MAX_DISTANCE
        finally:
            world.disconnect()

    def test_robot_starts_near_origin(self):
        world = PhysicsWorld(gui=False)
        try:
            pos = world.get_robot_position()
            assert abs(pos[0]) < 0.1
            assert abs(pos[1]) < 0.1
        finally:
            world.disconnect()

    def test_movement_in_all_horizontal_directions(self):
        world = PhysicsWorld(gui=False)
        try:
            for d in ["forward", "backward", "left", "right"]:
                world.reset()
                moved = world.move_robot(d)
                assert moved, f"Failed to move {d}"
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  PhysicsCamera Plugin
# ════════════════════════════════════════════

class TestPhysicsCameraUnit:
    def test_publishes_camera_3d_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=16, height=16)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            assert len(received) == 1
            ev = received[0]
            assert ev["type"] == "sensor.camera_3d"
            assert "depth" in ev["data"]
            assert "robot_position" in ev["data"]
        finally:
            world.disconnect()

    def test_depth_is_2d_matrix(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=16, height=16)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            depth = received[0]["data"]["depth"]
            assert isinstance(depth, list)
            assert len(depth) == 16
            assert len(depth[0]) == 16
            assert all(isinstance(v, float) for row in depth for v in row)
        finally:
            world.disconnect()

    def test_image_size_in_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=32, height=24)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.start()
            plugin.update()
            assert received[0]["data"]["image_size"] == [32, 24]
        finally:
            world.disconnect()


class TestPhysicsCameraEdgeCases:
    def test_no_event_before_start(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=8, height=8)
            received = []
            bus.subscribe("sensor.camera_3d", received.append)
            plugin.update()
            assert len(received) == 0
        finally:
            world.disconnect()


class TestPhysicsCameraMockData:
    def test_event_protocol_compliance(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsCamera(bus, world, width=8, height=8)
            plugin.start()
            plugin.update()
            ev = bus.get_history("sensor.camera_3d")[0]
            assert "timestamp" in ev
            assert "type" in ev
            assert "source" in ev
            assert "data" in ev
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  PhysicsDistance Plugin
# ════════════════════════════════════════════

class TestPhysicsDistanceUnit:
    def test_publishes_distance_3d_event(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            assert len(received) == 1
            ev = received[0]
            assert ev["type"] == "sensor.distance_3d"
            assert "distances" in ev["data"]
        finally:
            world.disconnect()

    def test_all_directions_present(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            distances = received[0]["data"]["distances"]
            for d in DIRECTIONS_3D:
                assert d in distances
                assert "distance" in distances[d]
                assert "raw_distance" in distances[d]
                assert "blocked" in distances[d]
        finally:
            world.disconnect()

    def test_obstacle_reduces_distance(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 0, 0.5)
            plugin = PhysicsDistance(bus, world)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.start()
            plugin.update()
            forward_dist = received[0]["data"]["distances"]["forward"]["distance"]
            assert forward_dist < _RAY_MAX_DISTANCE
        finally:
            world.disconnect()


class TestPhysicsDistanceEdgeCases:
    def test_no_event_before_start(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world)
            received = []
            bus.subscribe("sensor.distance_3d", received.append)
            plugin.update()
            assert len(received) == 0
        finally:
            world.disconnect()


class TestPhysicsDistanceMockData:
    def test_event_protocol_compliance(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            plugin = PhysicsDistance(bus, world)
            plugin.start()
            plugin.update()
            ev = bus.get_history("sensor.distance_3d")[0]
            assert "timestamp" in ev
            assert "type" in ev
            assert "source" in ev
            assert "data" in ev
            assert ev["data"]["noisy"] is False
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  PhysicsAgent
# ════════════════════════════════════════════

class TestPhysicsAgentUnit:
    def test_decide_without_data_picks_first(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            agent.decide()
            assert agent._chosen_action == PREFERRED_DIRECTIONS_3D[0]
        finally:
            world.disconnect()

    def test_act_publishes_move_3d(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            received = []
            bus.subscribe("action.move_3d", received.append)
            agent.decide()
            agent.act()
            assert len(received) == 1
            assert received[0]["type"] == "action.move_3d"
            assert "direction" in received[0]["data"]
        finally:
            world.disconnect()

    def test_cleanup_unsubscribes(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            agent.cleanup()
            received = []
            bus.subscribe("action.move_3d", received.append)
            bus.publish(_make_event("sensor.distance_3d", "test", {"distances": {}}))
            assert agent._latest_distance is None
        finally:
            world.disconnect()

    def test_visit_count_tracking(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            assert len(agent.visit_count) >= 1
        finally:
            world.disconnect()


class TestPhysicsAgentEdgeCases:
    def test_act_with_no_decision(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            agent._chosen_action = None
            received = []
            bus.subscribe("action.move_3d", received.append)
            agent.act()
            assert len(received) == 0
        finally:
            world.disconnect()

    def test_decide_all_blocked(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            all_blocked = {d: {"distance": 0.1, "blocked": True} for d in DIRECTIONS_3D}
            agent._latest_distance = _make_event(
                "sensor.distance_3d", "test",
                {"distances": all_blocked, "robot_position": [0, 0, 0.31]}
            )
            agent.decide()
            assert agent._chosen_action is None
        finally:
            world.disconnect()


class TestPhysicsAgentMockData:
    def test_full_pipeline_with_engine(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(3, 0, 0.5)
            plugin = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)
            engine = Engine(bus, enable_metrics=True)
            engine.register_plugin(plugin)
            engine.register_agent(agent)
            engine.run(steps=3)

            actions = bus.get_history("action.move_3d")
            assert len(actions) == 3
            for a in actions:
                assert "direction" in a["data"]
                assert "success" in a["data"]
                assert "new_position" in a["data"]
        finally:
            world.disconnect()

    def test_agent_moves_away_from_obstacle(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(1.5, 0, 0.5)
            plugin = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)
            engine = Engine(bus, enable_metrics=True)
            engine.register_plugin(plugin)
            engine.register_agent(agent)
            engine.run(steps=5)

            actions = bus.get_history("action.move_3d")
            successful = [a for a in actions if a["data"]["success"]]
            assert len(successful) >= 1
        finally:
            world.disconnect()

    def test_event_data_protocol(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            agent = PhysicsAgent(bus, world)
            agent.decide()
            agent.act()
            ev = bus.get_history("action.move_3d")[0]
            assert "timestamp" in ev
            assert ev["source"] == "physics_agent"
            assert "position_before" in ev["data"]
            assert "new_position" in ev["data"]
        finally:
            world.disconnect()


# ════════════════════════════════════════════
#  Integration — full 3D pipeline
# ════════════════════════════════════════════

class TestPhysicsIntegrationUnit:
    def test_engine_runs_full_3d_pipeline(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(3, 0, 0.5)
            world.place_obstacle(-2, 1, 0.5)

            camera = PhysicsCamera(bus, world, width=8, height=8)
            distance = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)

            engine = Engine(bus, enable_metrics=True)
            engine.register_plugin(distance)
            engine.register_plugin(camera)
            engine.register_agent(agent)

            engine.run(steps=5)

            camera_events = bus.get_history("sensor.camera_3d")
            distance_events = bus.get_history("sensor.distance_3d")
            action_events = bus.get_history("action.move_3d")
            metric_events = bus.get_history("engine.metrics")

            assert len(camera_events) == 5
            assert len(distance_events) == 5
            assert len(action_events) == 5
            assert len(metric_events) == 5
        finally:
            world.disconnect()


class TestPhysicsIntegrationEdgeCases:
    def test_empty_world_no_crashes(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            distance = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)
            engine = Engine(bus)
            engine.register_plugin(distance)
            engine.register_agent(agent)
            engine.run(steps=3)
        finally:
            world.disconnect()

    def test_world_reset_mid_pipeline(self):
        bus = EventBus()
        world = PhysicsWorld(gui=False)
        try:
            world.place_obstacle(2, 0, 0.5)
            distance = PhysicsDistance(bus, world)
            agent = PhysicsAgent(bus, world)
            engine = Engine(bus)
            engine.register_plugin(distance)
            engine.register_agent(agent)
            engine.run(steps=2)
            world.reset()
            pos = world.get_robot_position()
            assert abs(pos[0]) < 0.1
        finally:
            world.disconnect()


class TestPhysicsIntegrationMockData:
    def test_multiple_obstacles_pipeline(self):
        bus = EventBus(max_history=5000)
        world = PhysicsWorld(gui=False)
        try:
            for i in range(5):
                world.place_obstacle(2 + i, i - 2, 0.5)

            distance = PhysicsDistance(bus, world)
            camera = PhysicsCamera(bus, world, width=8, height=8)
            agent = PhysicsAgent(bus, world)

            engine = Engine(bus, enable_metrics=True)
            engine.register_plugin(distance)
            engine.register_plugin(camera)
            engine.register_agent(agent)
            engine.run(steps=5)

            assert engine.step_count == 5
            metrics = engine.get_metrics()
            assert metrics["steps"] == 5
            assert metrics["total_ms"] > 0
        finally:
            world.disconnect()
