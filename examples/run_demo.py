"""
run_demo.py — end-to-end demo that wires every module together
and runs the robot through a simple obstacle course.

Showcases: engine metrics, agent exploration memory,
           sensor noise, wildcard logging, random world factory.

Usage:
    python -m examples.run_demo
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, WILDCARD
from core.engine import Engine
from simulator.world import World, random_world
from simulator.sensor_sim import SensorSim
from plugins.mock_camera import MockCamera
from plugins.mock_distance import MockDistance
from agents.simple_agent import SimpleAgent


def build_manual_world() -> World:
    world = World(width=8, height=6)
    obstacles = [
        (2, 0), (2, 1), (2, 2),
        (4, 3), (4, 4), (4, 5),
        (6, 1), (6, 2),
    ]
    for x, y in obstacles:
        world.place_obstacle(x, y)
    world.set_robot_position(0, 0)
    return world


def run_scenario(title: str, world: World, noise: float, steps: int) -> None:
    bus = EventBus(max_history=5000)
    sensor_sim = SensorSim(world, noise_level=noise, seed=42)

    camera = MockCamera(bus, sensor_sim)
    distance = MockDistance(bus, sensor_sim)
    agent = SimpleAgent(bus, world)

    engine = Engine(bus, enable_metrics=True)
    engine.register_plugin(camera)
    engine.register_plugin(distance)
    engine.register_agent(agent)

    log = []
    bus.subscribe(WILDCARD, lambda e: log.append(e["type"]))

    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"World: {world.width}x{world.height}  |  Steps: {steps}  |  Noise: {noise}")
    print("Legend: R=robot  #=obstacle  .=free\n")
    print("--- Initial State ---")
    print(world.render())
    print()

    def on_step(step: int) -> None:
        pos = world.get_robot_position()
        print(f"--- Step {step} | Robot at {pos} ---")
        print(world.render())
        print()

    engine.run(steps=steps, on_step=on_step)

    actions = bus.get_history("action.move")
    moves = sum(1 for a in actions if a["data"]["success"])
    blocks = len(actions) - moves
    metrics = engine.get_metrics()
    visited = len(agent.visit_count)

    print("-" * 50)
    print(f"Actions: {len(actions)} | Moves: {moves} | Blocked: {blocks}")
    print(f"Unique cells visited: {visited}")
    print(f"Engine: {metrics['steps']} steps in {metrics['total_ms']:.2f}ms "
          f"(avg {metrics['avg_ms']:.3f}ms, max {metrics['max_ms']:.3f}ms)")
    print(f"Total bus events logged: {len(log)}")
    print()


def main() -> None:
    print("\n")
    run_scenario(
        title="Scenario 1 — Manual World (no noise)",
        world=build_manual_world(),
        noise=0.0,
        steps=20,
    )

    run_scenario(
        title="Scenario 2 — Random World (with sensor noise)",
        world=random_world(10, 8, obstacle_ratio=0.2, seed=7),
        noise=0.1,
        steps=25,
    )


if __name__ == "__main__":
    main()
