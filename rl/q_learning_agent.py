"""
QLearningAgent — simple tabular Q-learning agent for RobotEnv.

Demonstrates how to use the Gymnasium interface with a classic
RL algorithm.  Discretizes the continuous observation into
a hashable state for the Q-table.
"""

from __future__ import annotations

import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-learning with epsilon-greedy exploration.

    Observation is discretized by rounding each float to ``precision``
    decimal places to produce a hashable state key.
    """

    def __init__(
        self,
        n_actions: int = 6,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        precision: int = 1,
        seed: int | None = None,
    ) -> None:
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._precision = precision
        self._rng = random.Random(seed)
        self._q: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0] * self._n_actions
        )
        self._total_updates = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def q_table_size(self) -> int:
        return len(self._q)

    @property
    def total_updates(self) -> int:
        return self._total_updates

    def _discretize(self, obs) -> tuple:
        return tuple(round(float(v), self._precision) for v in obs)

    def select_action(self, obs) -> int:
        if self._rng.random() < self._epsilon:
            return self._rng.randint(0, self._n_actions - 1)
        state = self._discretize(obs)
        q_vals = self._q[state]
        return int(max(range(self._n_actions), key=lambda a: q_vals[a]))

    def update(self, obs, action: int, reward: float, next_obs, done: bool) -> float:
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        best_next = max(self._q[next_state]) if not done else 0.0
        td_target = reward + self._gamma * best_next
        td_error = td_target - self._q[state][action]
        self._q[state][action] += self._alpha * td_error

        self._total_updates += 1
        return td_error

    def decay_epsilon(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def get_q_values(self, obs) -> list[float]:
        state = self._discretize(obs)
        return list(self._q[state])
