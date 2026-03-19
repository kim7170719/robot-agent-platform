"""
RobotEnv — Gymnasium-compatible environment wrapping the IR-AAP
PhysicsWorld for reinforcement learning.

Observation: 6 distance readings + 3D robot position + 3D goal = 12 floats
Action:      Discrete(6) — forward/backward/left/right/up/down
Reward:      distance reduction toward goal, penalty for collision,
             bonus for reaching goal, small step penalty.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.event_bus import EventBus
from core.engine import Engine
from simulator.physics_world import PhysicsWorld, DIRECTIONS_3D
from plugins.physics_distance import PhysicsDistance
from plugins.goal_publisher import GoalPublisher

_DIRECTION_NAMES = list(DIRECTIONS_3D.keys())
_GOAL_REACHED_RADIUS = 0.8
_MAX_STEPS = 200
_DEFAULT_GOAL = (5.0, 0.0, 0.31)


class RobotEnv(gym.Env):
    """
    Gymnasium environment for the IR-AAP 3D robot simulation.

    Each ``step()`` calls ``engine.step()`` once (sensor update + physics)
    then reads distance sensor data to build the observation.
    """

    metadata = {"render_modes": ["human", "text"]}

    def __init__(
        self,
        goal: tuple[float, float, float] = _DEFAULT_GOAL,
        max_steps: int = _MAX_STEPS,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._goal = goal
        self._max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

        self._bus: EventBus | None = None
        self._engine: Engine | None = None
        self._world: PhysicsWorld | None = None
        self._dist_plugin: PhysicsDistance | None = None
        self._goal_pub: GoalPublisher | None = None
        self._step_count = 0
        self._latest_distances: dict = {}
        self._prev_goal_dist: float = 0.0

    def _setup(self) -> None:
        """Create fresh simulation objects."""
        self._bus = EventBus(max_history=0)
        self._world = PhysicsWorld(gui=False)
        self._dist_plugin = PhysicsDistance(self._bus, self._world)
        self._goal_pub = GoalPublisher(self._bus)
        self._goal_pub.set_target(*self._goal)

        self._engine = Engine(self._bus, enable_metrics=False)
        self._engine.register_plugin(self._dist_plugin)
        self._engine.register_plugin(self._goal_pub)

        self._bus.subscribe("sensor.distance_3d", self._on_distance)

        for plugin in self._engine._plugins:
            plugin.start()
        self._engine._running = True

    def _on_distance(self, event: dict) -> None:
        self._latest_distances = event.get("data", {}).get("distances", {})

    def _get_obs(self) -> np.ndarray:
        dists = []
        for name in _DIRECTION_NAMES:
            d = self._latest_distances.get(name, {}).get("distance", 20.0)
            dists.append(float(d))
        pos = self._world.get_robot_position()
        obs = np.array(
            dists + list(pos) + list(self._goal), dtype=np.float32,
        )
        return obs

    def _goal_distance(self) -> float:
        pos = self._world.get_robot_position()
        return math.dist(pos, self._goal)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.close()
        self._setup()
        self._step_count = 0
        self._latest_distances = {}

        self._dist_plugin.update()
        self._goal_pub.update()

        self._prev_goal_dist = self._goal_distance()
        obs = self._get_obs()
        info = {"goal_distance": self._prev_goal_dist}
        return obs, info

    def step(self, action: int):
        direction = _DIRECTION_NAMES[action]
        self._world.move_robot(direction)

        for plugin in self._engine._plugins:
            plugin.update()

        self._step_count += 1
        current_dist = self._goal_distance()

        reward = (self._prev_goal_dist - current_dist) * 10.0
        reward -= 0.1

        terminated = False
        if current_dist < _GOAL_REACHED_RADIUS:
            reward += 100.0
            terminated = True

        truncated = self._step_count >= self._max_steps

        self._prev_goal_dist = current_dist
        obs = self._get_obs()
        info = {
            "goal_distance": current_dist,
            "step": self._step_count,
            "direction": direction,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._world is None:
            return
        text = self._world.render()
        if self.render_mode == "human":
            print(text)
        return text

    def close(self):
        if self._engine is not None:
            self._engine._running = False
            self._engine = None
        if self._bus is not None:
            self._bus.clear()
            self._bus = None
        if self._world is not None:
            self._world.disconnect()
            self._world = None
        self._dist_plugin = None
        self._goal_pub = None
