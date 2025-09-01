"""
Training script for DRL project with evaluation, early stopping,
randomized episodes and logging.

The script expects a ``trading_env.TradingEnv`` implementation that accepts
``pandas.DataFrame`` data. If not available, a CartPole environment is used so
that the training loop can still execute for demonstration purposes.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv

try:  # pragma: no cover - optional import
    from trading_env import TradingEnv  # type: ignore
except Exception:  # pragma: no cover - fallback for demonstration
    TradingEnv = None


class RandomStartWrapper(gym.Wrapper):
    """Reset episodes from random start indices.

    The wrapped environment must accept ``start_index`` in ``reset`` and expose a
    ``current_step`` attribute counting how many timesteps have elapsed since the
    beginning of the dataset.
    """

    def __init__(self, env: gym.Env, max_start: int, episode_length: int | None = None):
        super().__init__(env)
        self.max_start = max_start
        self.episode_length = episode_length
        self.start_step = 0

    def reset(self, **kwargs):  # type: ignore[override]
        rng = getattr(self, "np_random", np.random.default_rng())
        self.start_step = int(rng.integers(0, self.max_start + 1))
        return self.env.reset(start_index=self.start_step, **kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if (
            self.episode_length is not None
            and (getattr(self.env, "current_step", 0) - self.start_step) >= self.episode_length
        ):
            truncated = True
            done = True
        return obs, reward, done, truncated, info


def make_env(data: pd.DataFrame, randomize: bool, episode_length: int | None = None):
    """Factory creating training or evaluation environments."""

    def _init() -> gym.Env:
        env = TradingEnv(data) if TradingEnv is not None else gym.make("CartPole-v1")
        if randomize and TradingEnv is not None:
            max_start = len(data) - 1
            env = RandomStartWrapper(env, max_start, episode_length)
        return env

    return _init


def main() -> None:
    # Load data - replace with real market data path.
    data_path = Path("market_data.csv")
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:  # fallback data for CartPole
        data = pd.DataFrame()

    split = int(0.8 * len(data)) if len(data) else 0
    train_data, eval_data = data.iloc[:split], data.iloc[split:]

    train_env = DummyVecEnv([make_env(train_data, randomize=True, episode_length=252)])
    eval_env = DummyVecEnv([make_env(eval_data, randomize=False)])

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(run_dir),
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=10
    )
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=1000,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        callback_after_eval=stop_callback,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
    )

    model.learn(
        total_timesteps=100_000,
        callback=[eval_callback, checkpoint_callback],
    )
    model.save(str(run_dir / "final_model"))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
