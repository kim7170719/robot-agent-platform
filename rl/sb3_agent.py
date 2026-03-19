"""
sb3_agent — Stable-Baselines3 DQN/PPO wrappers for RobotEnv.

Provides thin wrappers that handle the standard train→evaluate→save
lifecycle. Works out-of-the-box with the IR-AAP Gymnasium environment.
"""

from __future__ import annotations

from typing import Literal

from rl.robot_env import RobotEnv

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def _check_sb3() -> None:
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )


class SB3Agent:
    """
    Unified DQN / PPO agent powered by stable-baselines3.

    Parameters
    ----------
    algorithm : "DQN" | "PPO"
        Which RL algorithm to use.
    env_kwargs : dict | None
        Extra keyword args forwarded to ``RobotEnv()``.
    model_kwargs : dict | None
        Extra keyword args forwarded to the SB3 model constructor
        (e.g. ``learning_rate``, ``batch_size``, ``policy_kwargs``).
    """

    def __init__(
        self,
        algorithm: Literal["DQN", "PPO"] = "DQN",
        env_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        _check_sb3()
        self._algo_name = algorithm
        self._env_kwargs = env_kwargs or {}
        self._model_kwargs = model_kwargs or {}
        self._env: RobotEnv | None = None
        self._model = None
        self._total_timesteps_trained = 0

    @property
    def algorithm(self) -> str:
        return self._algo_name

    @property
    def total_timesteps_trained(self) -> int:
        return self._total_timesteps_trained

    @property
    def model(self):
        return self._model

    def _make_env(self) -> RobotEnv:
        env = RobotEnv(**self._env_kwargs)
        return env

    def build(self) -> None:
        """Create the environment and model (does not train)."""
        self._env = self._make_env()
        cls = DQN if self._algo_name == "DQN" else PPO

        defaults: dict = {
            "policy": "MlpPolicy",
            "verbose": 0,
        }
        if self._algo_name == "DQN":
            defaults.setdefault("buffer_size", 10_000)
            defaults.setdefault("learning_starts", 100)

        merged = {**defaults, **self._model_kwargs}
        self._model = cls(env=self._env, **merged)

    def train(self, total_timesteps: int = 1000) -> dict:
        """
        Train for *total_timesteps* steps.
        Returns a summary dict.
        """
        if self._model is None:
            self.build()
        self._model.learn(total_timesteps=total_timesteps)
        self._total_timesteps_trained += total_timesteps
        return {
            "algorithm": self._algo_name,
            "timesteps": total_timesteps,
            "total_trained": self._total_timesteps_trained,
        }

    def evaluate(self, n_eval_episodes: int = 5) -> dict:
        """
        Evaluate the current policy.
        Returns mean_reward, std_reward.
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build() or train() first.")
        eval_env = self._make_env()
        try:
            mean_reward, std_reward = evaluate_policy(
                self._model, eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
        finally:
            eval_env.close()
        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "n_episodes": n_eval_episodes,
        }

    def save(self, path: str) -> str:
        """Save the model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        self._model.save(path)
        return path

    def load(self, path: str) -> None:
        """Load a previously saved model."""
        if self._env is None:
            self._env = self._make_env()
        cls = DQN if self._algo_name == "DQN" else PPO
        self._model = cls.load(path, env=self._env)

    def predict(self, obs, deterministic: bool = True) -> int:
        """Predict action for a single observation."""
        if self._model is None:
            raise RuntimeError("No model loaded.")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def close(self) -> None:
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def get_status(self) -> dict:
        return {
            "algorithm": self._algo_name,
            "total_trained": self._total_timesteps_trained,
            "model_loaded": self._model is not None,
        }
