import argparse
import optuna
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "CartPole-v1"
POLICY_KWARGS = dict(net_arch=[256, 256])
TOTAL_TIMESTEPS = 5000


def optimize_ppo(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    n_steps = trial.suggest_int("n_steps", 16, 256, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    env = gym.make(ENV_ID)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()
    return mean_reward


def optimize_a2c(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    n_steps = trial.suggest_int("n_steps", 5, 256, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # sampled but unused
    env = gym.make(ENV_ID)
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()
    return mean_reward


def optimize_dqn(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_int("n_steps", 1, 8, log=True)
    env = gym.make(ENV_ID)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        train_freq=n_steps,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()
    return mean_reward


def run_study(name: str, objective, n_trials: int):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"{name} best reward: {study.best_value:.2f} with params {study.best_params}")
    return study.best_value, study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Trials per algorithm")
    parser.add_argument(
        "--timesteps", type=int, default=5000, help="Training timesteps per trial"
    )
    args = parser.parse_args()

    global TOTAL_TIMESTEPS
    TOTAL_TIMESTEPS = args.timesteps

    results = {}
    results["PPO"] = run_study("PPO", optimize_ppo, args.trials)
    results["A2C"] = run_study("A2C", optimize_a2c, args.trials)
    results["DQN"] = run_study("DQN", optimize_dqn, args.trials)

    best_algo = max(results.items(), key=lambda x: x[1][0])
    print(
        f"Best algorithm: {best_algo[0]} with reward {best_algo[1][0]:.2f} and params {best_algo[1][1]}"
    )


if __name__ == "__main__":
    main()
