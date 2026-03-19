"""
RL Training Demo — trains a Q-learning agent in the RobotEnv
Gymnasium environment for a few episodes and prints results.

Usage:
    python -m examples.run_rl_demo
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.robot_env import RobotEnv
from rl.q_learning_agent import QLearningAgent


def main(episodes: int = 30, max_steps: int = 80) -> None:
    env = RobotEnv(goal=(3.0, 0.0, 0.31), max_steps=max_steps)
    agent = QLearningAgent(n_actions=6, seed=42, epsilon_decay=0.98)

    print("=" * 60)
    print("  IR-AAP Reinforcement Learning Training Demo")
    print("=" * 60)

    best_reward = float("-inf")
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        agent.decay_epsilon()
        best_reward = max(best_reward, total_reward)

        status = "REACHED" if terminated else "timeout"
        print(
            f"  Episode {ep + 1:3d}/{episodes} | "
            f"Steps: {steps:3d} | "
            f"Reward: {total_reward:7.1f} | "
            f"Dist: {info['goal_distance']:5.2f} | "
            f"eps: {agent.epsilon:.3f} | "
            f"{status}"
        )

    env.close()

    print("\n" + "=" * 60)
    print("  Training Complete")
    print(f"  Best Reward: {best_reward:.1f}")
    print(f"  Q-table Size: {agent.q_table_size}")
    print(f"  Total Updates: {agent.total_updates}")
    print("=" * 60)


if __name__ == "__main__":
    main()
