"""
run_physics_demo.py — end-to-end 3D physics demo using PyBullet.

Runs the robot through a small obstacle field in headless mode by default.
Pass --gui to open the PyBullet 3D viewer (requires OpenGL).

Usage:
    python -m examples.run_physics_demo
    python -m examples.run_physics_demo --gui
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_bus import EventBus, WILDCARD
from core.engine import Engine
from simulator.physics_world import PhysicsWorld
from plugins.physics_camera import PhysicsCamera
from plugins.physics_distance import PhysicsDistance
from agents.physics_agent import PhysicsAgent


def build_physics_world(gui: bool) -> PhysicsWorld:
    world = PhysicsWorld(gui=gui, timestep=1 / 240)
    world.place_obstacle(3, 0, 0.5)
    world.place_obstacle(3, 2, 0.5)
    world.place_obstacle(-2, 1, 0.5)
    world.place_obstacle(1, 3, 0.5)
    world.place_obstacle(5, -1, 0.5)
    return world


def run_physics_demo(gui: bool = False, steps: int = 15) -> None:
    bus = EventBus(max_history=5000)
    world = build_physics_world(gui)

    camera_plugin = PhysicsCamera(bus, world, width=32, height=32)
    distance_plugin = PhysicsDistance(bus, world)
    agent = PhysicsAgent(bus, world)

    engine = Engine(bus, enable_metrics=True)
    engine.register_plugin(distance_plugin)
    engine.register_plugin(camera_plugin)
    engine.register_agent(agent)

    log: list[str] = []
    bus.subscribe(WILDCARD, lambda e: log.append(e["type"]))

    print("=" * 55)
    print("  3D Physics Simulator Demo (PyBullet)")
    print("=" * 55)
    print(f"Mode: {'GUI' if gui else 'Headless (DIRECT)'}")
    print(f"Steps: {steps}")
    print(f"Obstacles: {len(world._obstacle_ids)}")
    print()
    print("--- Initial State ---")
    print(world.render())
    print()

    def on_step(step: int) -> None:
        pos = world.get_robot_position()
        print(f"--- Step {step} | Robot at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) ---")

    engine.run(steps=steps, on_step=on_step)

    actions = bus.get_history("action.move_3d")
    moves = sum(1 for a in actions if a["data"]["success"])
    blocks = len(actions) - moves
    metrics = engine.get_metrics()
    visited = len(agent.visit_count)

    print()
    print("-" * 55)
    print(f"Actions: {len(actions)} | Moves: {moves} | Blocked: {blocks}")
    print(f"Unique 3D cells visited: {visited}")
    print(f"Engine: {metrics['steps']} steps in {metrics['total_ms']:.1f}ms "
          f"(avg {metrics['avg_ms']:.1f}ms, max {metrics['max_ms']:.1f}ms)")
    print(f"Total bus events logged: {len(log)}")
    print()
    print("--- Final State ---")
    print(world.render())
    print()

    world.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Physics Simulator Demo")
    parser.add_argument("--gui", action="store_true", help="Launch with PyBullet GUI")
    parser.add_argument("--steps", type=int, default=15, help="Number of simulation steps")
    args = parser.parse_args()
    run_physics_demo(gui=args.gui, steps=args.steps)


if __name__ == "__main__":
    main()
