"""
test_core.py — full test suite for the IR-AAP platform.

Per 測試規則.txt, every module includes:
  1. Unit tests
  2. Edge case tests
  3. Mock data tests

Run:  pytest tests/test_core.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, _make_event, _validate_event, WILDCARD
from core.engine import Engine
from simulator.world import World, DIRECTIONS, random_world
from simulator.sensor_sim import SensorSim
from plugins.base_plugin import BasePlugin
from plugins.mock_camera import MockCamera
from plugins.mock_distance import MockDistance
from agents.base_agent import BaseAgent
from agents.simple_agent import SimpleAgent


# ════════════════════════════════════════════
#  EventBus
# ════════════════════════════════════════════

class TestEventBusUnit:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", received.append)
        event = _make_event("test", "src", {"val": 1})
        bus.publish(event)
        assert len(received) == 1
        assert received[0]["data"]["val"] == 1

    def test_publish_to_correct_type_only(self):
        bus = EventBus()
        a_events, b_events = [], []
        bus.subscribe("a", a_events.append)
        bus.subscribe("b", b_events.append)
        bus.publish(_make_event("a", "src", {}))
        assert len(a_events) == 1
        assert len(b_events) == 0

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("x", received.append)
        bus.unsubscribe("x", received.append)
        bus.publish(_make_event("x", "src", {}))
        assert len(received) == 0

    def test_history(self):
        bus = EventBus()
        bus.publish(_make_event("a", "src", {}))
        bus.publish(_make_event("b", "src", {}))
        assert len(bus.get_history()) == 2
        assert len(bus.get_history("a")) == 1

    def test_clear(self):
        bus = EventBus()
        bus.subscribe("x", lambda e: None)
        bus.publish(_make_event("x", "src", {}))
        bus.clear()
        assert bus.get_history() == []


class TestEventBusEdgeCases:
    def test_publish_no_subscribers(self):
        bus = EventBus()
        bus.publish(_make_event("orphan", "src", {}))
        assert len(bus.get_history()) == 1

    def test_invalid_event_not_dict(self):
        bus = EventBus()
        with pytest.raises(TypeError):
            bus.publish("not a dict")

    def test_invalid_event_missing_fields(self):
        bus = EventBus()
        with pytest.raises(ValueError):
            bus.publish({"timestamp": 0})

    def test_subscribe_non_callable(self):
        bus = EventBus()
        with pytest.raises(TypeError):
            bus.subscribe("x", "not_callable")

    def test_unsubscribe_nonexistent(self):
        bus = EventBus()
        bus.unsubscribe("nope", lambda e: None)  # should not raise


class TestEventBusMockData:
    def test_protocol_compliant_event(self):
        event = _make_event("sensor.camera", "cam_1", {"frame": [0, 1, 2]})
        _validate_event(event)
        assert event["type"] == "sensor.camera"
        assert event["source"] == "cam_1"
        assert isinstance(event["timestamp"], int)
        assert isinstance(event["data"], dict)

    def test_multiple_subscribers_same_type(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("s", r1.append)
        bus.subscribe("s", r2.append)
        bus.publish(_make_event("s", "src", {"v": 42}))
        assert len(r1) == 1 and len(r2) == 1


# ════════════════════════════════════════════
#  World
# ════════════════════════════════════════════

class TestWorldUnit:
    def test_initial_state(self):
        w = World(5, 5)
        assert w.get_robot_position() == (0, 0)
        assert w.get_obstacle_map() == [[0]*5 for _ in range(5)]

    def test_place_and_check_obstacle(self):
        w = World(5, 5)
        w.place_obstacle(3, 3)
        assert w.is_obstacle(3, 3)
        assert not w.is_obstacle(0, 0)

    def test_remove_obstacle(self):
        w = World(5, 5)
        w.place_obstacle(1, 1)
        w.remove_obstacle(1, 1)
        assert not w.is_obstacle(1, 1)

    def test_move_robot(self):
        w = World(5, 5)
        assert w.move_robot("right")
        assert w.get_robot_position() == (1, 0)

    def test_move_blocked_by_obstacle(self):
        w = World(5, 5)
        w.place_obstacle(1, 0)
        assert not w.move_robot("right")
        assert w.get_robot_position() == (0, 0)

    def test_render(self):
        w = World(3, 3)
        w.place_obstacle(2, 2)
        text = w.render()
        assert "R" in text
        assert "#" in text


class TestWorldEdgeCases:
    def test_move_out_of_bounds(self):
        w = World(3, 3)
        assert not w.move_robot("up")
        assert not w.move_robot("left")

    def test_invalid_direction(self):
        w = World(3, 3)
        with pytest.raises(ValueError):
            w.move_robot("diagonal")

    def test_place_obstacle_on_robot(self):
        w = World(3, 3)
        with pytest.raises(ValueError):
            w.place_obstacle(0, 0)

    def test_place_obstacle_out_of_bounds(self):
        w = World(3, 3)
        with pytest.raises(IndexError):
            w.place_obstacle(10, 10)

    def test_out_of_bounds_is_obstacle(self):
        w = World(3, 3)
        assert w.is_obstacle(-1, 0)
        assert w.is_obstacle(0, 100)

    def test_minimum_world_size(self):
        w = World(1, 1)
        assert w.get_robot_position() == (0, 0)
        assert not w.move_robot("right")

    def test_invalid_world_size(self):
        with pytest.raises(ValueError):
            World(0, 5)


class TestWorldMockData:
    def test_surroundings_open(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        surr = w.get_surroundings(2, 2, radius=1)
        for d in DIRECTIONS:
            assert not surr[d]["blocked"]

    def test_surroundings_blocked(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        w.place_obstacle(3, 2)
        surr = w.get_surroundings(2, 2, radius=3)
        assert surr["right"]["blocked"]
        assert surr["right"]["distance"] == 1

    def test_obstacle_map_is_copy(self):
        w = World(3, 3)
        grid = w.get_obstacle_map()
        grid[0][0] = 99
        assert w.get_obstacle_map()[0][0] == 0


# ════════════════════════════════════════════
#  SensorSim
# ════════════════════════════════════════════

class TestSensorSimUnit:
    def test_read_camera_structure(self):
        w = World(5, 5)
        sim = SensorSim(w)
        event = sim.read_camera()
        assert event["type"] == "sensor.camera"
        assert "field_of_view" in event["data"]

    def test_read_distance_structure(self):
        w = World(5, 5)
        sim = SensorSim(w)
        event = sim.read_distance()
        assert event["type"] == "sensor.distance"
        assert "distances" in event["data"]


class TestSensorSimEdgeCases:
    def test_camera_at_corner(self):
        w = World(3, 3)
        sim = SensorSim(w)
        event = sim.read_camera()
        fov = event["data"]["field_of_view"]
        assert fov["-1,-1"] == 1  # out-of-bounds → obstacle

    def test_distance_surrounded(self):
        w = World(3, 3)
        w.set_robot_position(1, 1)
        for x, y in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            w.place_obstacle(x, y)
        sim = SensorSim(w)
        event = sim.read_distance()
        for d in event["data"]["distances"].values():
            assert d["blocked"]


class TestSensorSimMockData:
    def test_protocol_compliance(self):
        w = World(5, 5)
        sim = SensorSim(w)
        for reader in [sim.read_camera, sim.read_distance]:
            event = reader()
            _validate_event(event)
            assert isinstance(event["timestamp"], int)
            assert isinstance(event["data"], dict)


# ════════════════════════════════════════════
#  MockCamera Plugin
# ════════════════════════════════════════════

class TestMockCameraUnit:
    def test_publishes_event(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        cam = MockCamera(bus, sim)
        cam.start()
        cam.update()
        history = bus.get_history("sensor.camera")
        assert len(history) == 1

    def test_does_not_publish_before_start(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        cam.update()
        assert len(bus.get_history("sensor.camera")) == 0

    def test_stop_prevents_further_updates(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        cam.start()
        cam.stop()
        cam.update()
        assert len(bus.get_history("sensor.camera")) == 0


class TestMockCameraEdgeCases:
    def test_multiple_updates(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        cam.start()
        for _ in range(5):
            cam.update()
        assert len(bus.get_history("sensor.camera")) == 5


class TestMockCameraMockData:
    def test_event_contains_fov(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(5, 5)))
        cam.start()
        cam.update()
        event = bus.get_history("sensor.camera")[0]
        assert "field_of_view" in event["data"]
        assert event["source"] == "sensor_sim"


# ════════════════════════════════════════════
#  MockDistance Plugin
# ════════════════════════════════════════════

class TestMockDistanceUnit:
    def test_publishes_event(self):
        bus = EventBus()
        dist = MockDistance(bus, SensorSim(World(5, 5)))
        dist.start()
        dist.update()
        assert len(bus.get_history("sensor.distance")) == 1

    def test_does_not_publish_before_start(self):
        bus = EventBus()
        dist = MockDistance(bus, SensorSim(World(3, 3)))
        dist.update()
        assert len(bus.get_history("sensor.distance")) == 0


class TestMockDistanceEdgeCases:
    def test_restart_cycle(self):
        bus = EventBus()
        dist = MockDistance(bus, SensorSim(World(3, 3)))
        dist.start()
        dist.update()
        dist.stop()
        dist.update()  # should not publish
        dist.start()
        dist.update()
        assert len(bus.get_history("sensor.distance")) == 2


class TestMockDistanceMockData:
    def test_event_contains_distances(self):
        bus = EventBus()
        dist = MockDistance(bus, SensorSim(World(5, 5)))
        dist.start()
        dist.update()
        event = bus.get_history("sensor.distance")[0]
        assert "distances" in event["data"]


# ════════════════════════════════════════════
#  SimpleAgent
# ════════════════════════════════════════════

class TestSimpleAgentUnit:
    def _make_env(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        dist = MockDistance(bus, sim)
        agent = SimpleAgent(bus, w)
        return bus, w, sim, dist, agent

    def test_moves_in_open_space(self):
        bus, w, sim, dist, agent = self._make_env()
        w.set_robot_position(2, 2)
        dist.start()
        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()
        assert w.get_robot_position() != (2, 2)

    def test_publishes_action_event(self):
        bus, w, sim, dist, agent = self._make_env()
        dist.start()
        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()
        actions = bus.get_history("action.move")
        assert len(actions) == 1
        assert "direction" in actions[0]["data"]


class TestSimpleAgentEdgeCases:
    def test_no_sensor_data_still_acts(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        agent.perceive()
        agent.decide()
        agent.act()
        assert w.get_robot_position() == (1, 0)

    def test_fully_surrounded(self):
        bus = EventBus()
        w = World(3, 3)
        w.set_robot_position(1, 1)
        for x, y in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            w.place_obstacle(x, y)
        sim = SensorSim(w)
        dist = MockDistance(bus, sim)
        agent = SimpleAgent(bus, w)
        dist.start()
        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()
        assert w.get_robot_position() == (1, 1)


class TestSimpleAgentMockData:
    def test_action_event_protocol(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        dist = MockDistance(bus, sim)
        agent = SimpleAgent(bus, w)
        dist.start()
        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()
        action = bus.get_history("action.move")[0]
        _validate_event(action)
        assert action["data"]["success"] in (True, False)
        assert isinstance(action["data"]["new_position"], list)


# ════════════════════════════════════════════
#  Engine (integration)
# ════════════════════════════════════════════

class TestEngineUnit:
    def test_single_step(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        cam = MockCamera(bus, sim)
        dist = MockDistance(bus, sim)
        agent = SimpleAgent(bus, w)
        engine = Engine(bus)
        engine.register_plugin(cam)
        engine.register_plugin(dist)
        engine.register_agent(agent)
        engine.run(steps=1)
        assert engine.step_count == 1
        assert len(bus.get_history("action.move")) == 1

    def test_multi_step(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        engine = Engine(bus)
        engine.register_plugin(MockCamera(bus, sim))
        engine.register_plugin(MockDistance(bus, sim))
        engine.register_agent(SimpleAgent(bus, w))
        engine.run(steps=3)
        assert engine.step_count == 3


class TestEngineEdgeCases:
    def test_run_zero_steps(self):
        bus = EventBus()
        engine = Engine(bus)
        engine.run(steps=0)
        assert engine.step_count == 0

    def test_no_plugins_no_agents(self):
        bus = EventBus()
        engine = Engine(bus)
        engine.run(steps=1)
        assert engine.step_count == 1

    def test_on_step_callback(self):
        bus = EventBus()
        engine = Engine(bus)
        calls = []
        engine.run(steps=3, on_step=calls.append)
        assert calls == [1, 2, 3]


class TestEngineMockData:
    def test_full_pipeline(self):
        bus = EventBus()
        w = World(8, 6)
        w.place_obstacle(2, 0)
        w.place_obstacle(2, 1)
        sim = SensorSim(w)
        engine = Engine(bus)
        engine.register_plugin(MockCamera(bus, sim))
        engine.register_plugin(MockDistance(bus, sim))
        engine.register_agent(SimpleAgent(bus, w))
        engine.run(steps=5)

        camera_events = bus.get_history("sensor.camera")
        distance_events = bus.get_history("sensor.distance")
        action_events = bus.get_history("action.move")
        assert len(camera_events) == 5
        assert len(distance_events) == 5
        assert len(action_events) == 5
        for e in action_events:
            _validate_event(e)


# ════════════════════════════════════════════
#  EventBus — Wildcard & History Cap (new)
# ════════════════════════════════════════════

class TestEventBusWildcardUnit:
    def test_wildcard_receives_all_events(self):
        bus = EventBus()
        received = []
        bus.subscribe(WILDCARD, received.append)
        bus.publish(_make_event("a", "src", {}))
        bus.publish(_make_event("b", "src", {}))
        assert len(received) == 2

    def test_wildcard_and_typed_both_fire(self):
        bus = EventBus()
        typed, wild = [], []
        bus.subscribe("x", typed.append)
        bus.subscribe(WILDCARD, wild.append)
        bus.publish(_make_event("x", "src", {}))
        assert len(typed) == 1
        assert len(wild) == 1

    def test_publish_count_tracks(self):
        bus = EventBus()
        for i in range(10):
            bus.publish(_make_event("t", "src", {"i": i}))
        assert bus.publish_count == 10


class TestEventBusWildcardEdgeCases:
    def test_wildcard_only_no_typed(self):
        bus = EventBus()
        received = []
        bus.subscribe(WILDCARD, received.append)
        bus.publish(_make_event("unregistered_type", "src", {}))
        assert len(received) == 1

    def test_unsubscribe_wildcard(self):
        bus = EventBus()
        received = []
        bus.subscribe(WILDCARD, received.append)
        bus.unsubscribe(WILDCARD, received.append)
        bus.publish(_make_event("a", "src", {}))
        assert len(received) == 0

    def test_history_cap_enforced(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            bus.publish(_make_event("t", "src", {"i": i}))
        history = bus.get_history()
        assert len(history) == 5
        assert history[0]["data"]["i"] == 5


class TestEventBusWildcardMockData:
    def test_clear_resets_publish_count(self):
        bus = EventBus()
        bus.publish(_make_event("a", "src", {}))
        bus.clear()
        assert bus.publish_count == 0
        assert bus.get_history() == []

    def test_default_max_history_is_large(self):
        bus = EventBus()
        assert bus.max_history == 10_000


# ════════════════════════════════════════════
#  Engine — Metrics (new)
# ════════════════════════════════════════════

class TestEngineMetricsUnit:
    def test_metrics_event_published(self):
        bus = EventBus()
        engine = Engine(bus, enable_metrics=True)
        engine.run(steps=3)
        metrics_events = bus.get_history("engine.metrics")
        assert len(metrics_events) == 3
        assert metrics_events[0]["data"]["step"] == 1

    def test_get_metrics_summary(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        engine = Engine(bus)
        engine.register_plugin(MockDistance(bus, sim))
        engine.register_agent(SimpleAgent(bus, w))
        engine.run(steps=5)
        m = engine.get_metrics()
        assert m["steps"] == 5
        assert m["total_ms"] >= 0
        assert m["avg_ms"] >= 0
        assert m["min_ms"] <= m["max_ms"]

    def test_step_times_recorded(self):
        bus = EventBus()
        engine = Engine(bus)
        engine.run(steps=3)
        assert len(engine.step_times) == 3
        assert all(t >= 0 for t in engine.step_times)


class TestEngineMetricsEdgeCases:
    def test_metrics_disabled(self):
        bus = EventBus()
        engine = Engine(bus, enable_metrics=False)
        engine.run(steps=3)
        assert len(bus.get_history("engine.metrics")) == 0
        assert engine.step_count == 3

    def test_metrics_zero_steps(self):
        bus = EventBus()
        engine = Engine(bus)
        m = engine.get_metrics()
        assert m == {"steps": 0, "total_ms": 0.0, "avg_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}


class TestEngineMetricsMockData:
    def test_metrics_event_protocol(self):
        bus = EventBus()
        engine = Engine(bus)
        engine.run(steps=1)
        event = bus.get_history("engine.metrics")[0]
        _validate_event(event)
        assert "elapsed_ms" in event["data"]
        assert event["source"] == "engine"


# ════════════════════════════════════════════
#  SimpleAgent — Exploration Memory (new)
# ════════════════════════════════════════════

class TestSimpleAgentMemoryUnit:
    def test_visit_count_tracks(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        assert agent.visit_count[(0, 0)] == 1

    def test_avoids_revisit(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        dist = MockDistance(bus, sim)
        agent = SimpleAgent(bus, w)
        dist.start()

        positions = [w.get_robot_position()]
        for _ in range(8):
            dist.update()
            agent.perceive()
            agent.decide()
            agent.act()
            positions.append(w.get_robot_position())

        unique = set(positions)
        assert len(unique) > 4


class TestSimpleAgentMemoryEdgeCases:
    def test_memory_not_reset_between_steps(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        sim = SensorSim(w)
        dist = MockDistance(bus, sim)
        dist.start()

        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()

        dist.update()
        agent.perceive()
        agent.decide()
        agent.act()

        total_visits = sum(agent.visit_count.values())
        assert total_visits == 3  # initial + 2 moves


class TestSimpleAgentMemoryMockData:
    def test_visit_count_is_copy(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        vc = agent.visit_count
        vc[(99, 99)] = 999
        assert (99, 99) not in agent.visit_count


# ════════════════════════════════════════════
#  World — random_world factory (new)
# ════════════════════════════════════════════

class TestRandomWorldUnit:
    def test_creates_world(self):
        w = random_world(10, 10, obstacle_ratio=0.3, seed=42)
        assert w.width == 10
        assert w.height == 10

    def test_start_position_is_free(self):
        for seed in range(20):
            w = random_world(8, 8, obstacle_ratio=0.4, seed=seed)
            assert not w.is_obstacle(0, 0)

    def test_deterministic_with_seed(self):
        w1 = random_world(6, 6, seed=123)
        w2 = random_world(6, 6, seed=123)
        assert w1.get_obstacle_map() == w2.get_obstacle_map()


class TestRandomWorldEdgeCases:
    def test_zero_obstacles(self):
        w = random_world(5, 5, obstacle_ratio=0.0)
        flat = [cell for row in w.get_obstacle_map() for cell in row]
        assert sum(flat) == 0

    def test_high_obstacle_ratio(self):
        w = random_world(5, 5, obstacle_ratio=0.99, seed=7)
        assert not w.is_obstacle(0, 0)

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            random_world(5, 5, obstacle_ratio=1.0)
        with pytest.raises(ValueError):
            random_world(5, 5, obstacle_ratio=-0.1)


class TestRandomWorldMockData:
    def test_different_seeds_different_worlds(self):
        w1 = random_world(8, 8, seed=1)
        w2 = random_world(8, 8, seed=2)
        assert w1.get_obstacle_map() != w2.get_obstacle_map()


# ════════════════════════════════════════════
#  World — reset (new)
# ════════════════════════════════════════════

class TestWorldResetUnit:
    def test_reset_clears_obstacles(self):
        w = World(5, 5)
        w.place_obstacle(2, 2)
        w.set_robot_position(3, 3)
        w.reset()
        assert w.get_robot_position() == (0, 0)
        assert not w.is_obstacle(2, 2)


class TestWorldResetEdgeCases:
    def test_reset_after_moves(self):
        w = World(5, 5)
        w.move_robot("right")
        w.move_robot("down")
        w.reset()
        assert w.get_robot_position() == (0, 0)


class TestWorldResetMockData:
    def test_reset_produces_clean_grid(self):
        w = random_world(5, 5, obstacle_ratio=0.5, seed=1)
        w.reset()
        flat = [cell for row in w.get_obstacle_map() for cell in row]
        assert sum(flat) == 0


# ════════════════════════════════════════════
#  SensorSim — Noise (new)
# ════════════════════════════════════════════

class TestSensorSimNoiseUnit:
    def test_no_noise_by_default(self):
        sim = SensorSim(World(5, 5))
        assert sim.noise_level == 0.0
        event = sim.read_camera()
        assert event["data"]["noisy"] is False

    def test_noise_flag_in_event(self):
        sim = SensorSim(World(5, 5), noise_level=0.1)
        cam = sim.read_camera()
        dist = sim.read_distance()
        assert cam["data"]["noisy"] is True
        assert dist["data"]["noisy"] is True

    def test_noise_level_setter(self):
        sim = SensorSim(World(5, 5))
        sim.noise_level = 0.3
        assert sim.noise_level == 0.3


class TestSensorSimNoiseEdgeCases:
    def test_negative_noise_clamped(self):
        sim = SensorSim(World(5, 5), noise_level=-1.0)
        assert sim.noise_level == 0.0

    def test_high_noise_distance_still_positive(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        sim = SensorSim(w, noise_level=2.0, seed=42)
        event = sim.read_distance()
        for info in event["data"]["distances"].values():
            assert info["distance"] >= 1

    def test_noise_setter_clamps_negative(self):
        sim = SensorSim(World(5, 5))
        sim.noise_level = -5.0
        assert sim.noise_level == 0.0


class TestSensorSimNoiseMockData:
    def test_noisy_distance_has_raw(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        sim = SensorSim(w, noise_level=0.5, seed=10)
        event = sim.read_distance()
        for info in event["data"]["distances"].values():
            assert "raw_distance" in info

    def test_deterministic_noise_with_seed(self):
        w = World(5, 5)
        s1 = SensorSim(w, noise_level=0.3, seed=99)
        s2 = SensorSim(w, noise_level=0.3, seed=99)
        assert s1.read_camera() == s2.read_camera()


# ════════════════════════════════════════════
#  EventBus — Exception Isolation (fix)
# ════════════════════════════════════════════

class TestEventBusExceptionUnit:
    def test_bad_subscriber_does_not_block_others(self):
        bus = EventBus()
        received = []

        def bad_cb(e):
            raise RuntimeError("boom")

        bus.subscribe("x", bad_cb)
        bus.subscribe("x", received.append)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            bus.publish(_make_event("x", "src", {}))
        assert len(received) == 1

    def test_bad_wildcard_does_not_block_typed(self):
        bus = EventBus()
        typed = []
        bus.subscribe("x", typed.append)
        bus.subscribe(WILDCARD, lambda e: 1 / 0)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            bus.publish(_make_event("x", "src", {}))
        assert len(typed) == 1


class TestEventBusExceptionEdgeCases:
    def test_warning_emitted_on_failure(self):
        bus = EventBus()
        bus.subscribe("x", lambda e: 1 / 0)
        import warnings as _w
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            bus.publish(_make_event("x", "src", {}))
        assert any("ZeroDivisionError" in str(warning.message) for warning in w)


class TestEventBusExceptionMockData:
    def test_history_recorded_despite_failure(self):
        bus = EventBus()
        bus.subscribe("x", lambda e: 1 / 0)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            bus.publish(_make_event("x", "src", {"v": 1}))
        assert len(bus.get_history("x")) == 1


# ════════════════════════════════════════════
#  EventBus — max_history=0 (fix)
# ════════════════════════════════════════════

class TestEventBusNoHistoryUnit:
    def test_no_history_when_zero(self):
        bus = EventBus(max_history=0)
        bus.publish(_make_event("a", "src", {}))
        assert bus.get_history() == []
        assert bus.publish_count == 1

    def test_subscribers_still_fire_with_no_history(self):
        bus = EventBus(max_history=0)
        received = []
        bus.subscribe("a", received.append)
        bus.publish(_make_event("a", "src", {}))
        assert len(received) == 1


class TestEventBusNoHistoryEdgeCases:
    def test_type_index_empty_when_no_history(self):
        bus = EventBus(max_history=0)
        bus.publish(_make_event("a", "src", {}))
        assert bus.get_history("a") == []


class TestEventBusNoHistoryMockData:
    def test_clear_works_on_no_history_bus(self):
        bus = EventBus(max_history=0)
        bus.publish(_make_event("a", "src", {}))
        bus.clear()
        assert bus.publish_count == 0


# ════════════════════════════════════════════
#  World — get_surroundings radius=0 (fix)
# ════════════════════════════════════════════

class TestWorldSurroundingsRadiusZeroUnit:
    def test_radius_zero_returns_safe_default(self):
        w = World(5, 5)
        surr = w.get_surroundings(2, 2, radius=0)
        for d in DIRECTIONS:
            assert not surr[d]["blocked"]
            assert surr[d]["distance"] == 1


class TestWorldSurroundingsRadiusZeroEdgeCases:
    def test_radius_zero_at_corner(self):
        w = World(3, 3)
        surr = w.get_surroundings(0, 0, radius=0)
        assert isinstance(surr, dict)
        assert len(surr) == 4


class TestWorldSurroundingsRadiusZeroMockData:
    def test_radius_zero_structure_matches_normal(self):
        w = World(5, 5)
        r0 = w.get_surroundings(2, 2, radius=0)
        r1 = w.get_surroundings(2, 2, radius=1)
        assert set(r0.keys()) == set(r1.keys())
        for d in r0:
            assert "blocked" in r0[d]
            assert "distance" in r0[d]


# ════════════════════════════════════════════
#  Agent — cleanup (fix)
# ════════════════════════════════════════════

class TestAgentCleanupUnit:
    def test_cleanup_unsubscribes(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        agent.cleanup()
        bus.publish(_make_event("sensor.distance", "test", {"distances": {}}))
        assert agent._latest_distance is None


class TestAgentCleanupEdgeCases:
    def test_double_cleanup_safe(self):
        bus = EventBus()
        w = World(5, 5)
        agent = SimpleAgent(bus, w)
        agent.cleanup()
        agent.cleanup()


class TestAgentCleanupMockData:
    def test_engine_stop_calls_cleanup(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)
        agent = SimpleAgent(bus, w)
        engine = Engine(bus)
        engine.register_plugin(MockDistance(bus, sim))
        engine.register_agent(agent)
        engine.run(steps=1)
        bus.publish(_make_event("sensor.distance", "test", {"distances": {}}))
        assert agent._latest_distance is not None  # from the run
        old = agent._latest_distance
        bus.publish(_make_event("sensor.distance", "test2", {"distances": {}}))
        assert agent._latest_distance is old  # cleanup prevented update


# ════════════════════════════════════════════
#  BasePlugin — active lifecycle (fix)
# ════════════════════════════════════════════

class TestBasePluginActiveUnit:
    def test_active_property(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        assert not cam.active
        cam.start()
        assert cam.active
        cam.stop()
        assert not cam.active


class TestBasePluginActiveEdgeCases:
    def test_update_before_start_no_event(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        cam.update()
        assert len(bus.get_history()) == 0


class TestBasePluginActiveMockData:
    def test_on_update_called_when_active(self):
        bus = EventBus()
        cam = MockCamera(bus, SensorSim(World(3, 3)))
        cam.start()
        cam.update()
        assert len(bus.get_history("sensor.camera")) == 1


# ════════════════════════════════════════════
#  Engine — Exception Isolation (fix)
# ════════════════════════════════════════════

class TestEngineExceptionUnit:
    def test_bad_plugin_does_not_stop_step(self):
        bus = EventBus()

        class BadPlugin(BasePlugin):
            def on_update(self):
                raise RuntimeError("plugin fail")

        engine = Engine(bus, enable_metrics=False)
        engine.register_plugin(BadPlugin(bus, "bad"))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            engine.run(steps=1)
        assert engine.step_count == 1


class TestEngineExceptionEdgeCases:
    def test_bad_agent_does_not_stop_step(self):
        bus = EventBus()

        class BadAgent(BaseAgent):
            def perceive(self):
                raise RuntimeError("agent fail")
            def decide(self):
                pass
            def act(self):
                pass

        engine = Engine(bus, enable_metrics=False)
        engine.register_agent(BadAgent(bus, "bad"))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            engine.run(steps=1)
        assert engine.step_count == 1


class TestEngineExceptionMockData:
    def test_good_plugin_runs_despite_bad_one(self):
        bus = EventBus()
        w = World(5, 5)
        sim = SensorSim(w)

        class BadPlugin(BasePlugin):
            def on_update(self):
                raise RuntimeError("fail")

        engine = Engine(bus, enable_metrics=False)
        engine.register_plugin(BadPlugin(bus, "bad"))
        engine.register_plugin(MockCamera(bus, sim))
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            engine.run(steps=1)
        assert len(bus.get_history("sensor.camera")) == 1


# ════════════════════════════════════════════
#  SensorSim — Unified Distance Structure (fix)
# ════════════════════════════════════════════

class TestSensorSimUnifiedUnit:
    def test_raw_distance_always_present(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        sim_clean = SensorSim(w, noise_level=0.0)
        sim_noisy = SensorSim(w, noise_level=0.5, seed=1)
        for sim in [sim_clean, sim_noisy]:
            event = sim.read_distance()
            for info in event["data"]["distances"].values():
                assert "raw_distance" in info


class TestSensorSimUnifiedEdgeCases:
    def test_raw_equals_distance_when_no_noise(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        sim = SensorSim(w)
        event = sim.read_distance()
        for info in event["data"]["distances"].values():
            assert info["raw_distance"] == info["distance"]


class TestSensorSimUnifiedMockData:
    def test_structure_identical_keys(self):
        w = World(5, 5)
        w.set_robot_position(2, 2)
        clean = SensorSim(w, noise_level=0.0).read_distance()
        noisy = SensorSim(w, noise_level=0.5, seed=1).read_distance()
        for d in DIRECTIONS:
            assert set(clean["data"]["distances"][d].keys()) == set(noisy["data"]["distances"][d].keys())
