import gymnasium as gym
from cv2.ml import TRAIN_ERROR
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import torch
import os
from tqdm import tqdm
from datetime import datetime
from envopenaigym import AutonomousCarGymnasiumEnv


class TqdmCallback(EvalCallback):
    def __init__(self, pbar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbar = pbar

    def _on_step(self) -> bool:
        # Update progress bar
        self.pbar.update(1)
        return super()._on_step()


def make_env(rank: int, seed: int = 0) -> callable:
    def _init() -> gym.Env:
        env = AutonomousCarGymnasiumEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# def custom_training_loop(env, model, total_timesteps: int, batch_size: int = 2048,
#                          reward_window: int = 1000, model_dir: str = None, log_dir: str = None):
#     """
#     Custom training loop that tracks rolling average rewards during training
#
#     Args:
#         env: Vectorized training environment
#         model: Initialized RL model (PPO, SAC, or DDPG)
#         total_timesteps: Total number of timesteps to train
#         batch_size: Number of steps to collect before updating the policy
#         reward_window: Number of recent rewards to average over
#         model_dir: Directory to save best models
#         log_dir: Directory for logging
#     """
#     # Initialize tracking variables
#     timesteps_so_far = 0
#     best_mean_reward = -np.inf
#
#     # Use deque for efficient rolling window calculation
#     from collections import deque
#     recent_rewards = deque(maxlen=reward_window)
#     current_episode_reward = 0
#
#     # Ensure logger is initialized
#     if not hasattr(model, '_logger'):
#         from stable_baselines3.common.logger import configure
#         model._logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
#
#     # Get initial observation
#     obs = env.reset()
#
#     # Progress bar
#     pbar = tqdm(total=total_timesteps, desc=f"Training {model.__class__.__name__}")
#
#     while timesteps_so_far < total_timesteps:
#         # Get action from policy
#         action, _ = model.predict(obs, deterministic=True)
#
#         # Take action in environment
#         next_obs, reward, done, info = env.step(action)
#         print(done)
#
#         # Update tracking
#         current_episode_reward += reward[0]
#         timesteps_so_far += 1
#         pbar.update(1)
#
#         # Store transition in buffer
#         if isinstance(model, SAC):
#             model.replay_buffer.add(
#                 obs,
#                 next_obs,
#                 action,
#                 reward,
#                 done,
#                 info
#             )
#
#         # Move to next observation
#         obs = next_obs
#
#         # Handle episode termination
#         if done[0]:
#             recent_rewards.append(current_episode_reward)
#             current_episode_reward = 0
#             obs = env.reset()
#
#             # Calculate and log average reward
#             if len(recent_rewards) > 0:
#                 mean_reward = np.mean(recent_rewards)
#                 print(f"\nTimestep: {timesteps_so_far}")
#                 print(f"Average reward over last {len(recent_rewards)} episodes: {mean_reward:.2f}")
#
#                 # Log to tensorboard
#                 if hasattr(model, '_logger'):
#                     model._logger.record("train/mean_reward", mean_reward)
#                     model._logger.dump(timesteps_so_far)
#
#                 # Save best model
#                 if mean_reward > best_mean_reward and model_dir is not None:
#                     best_mean_reward = mean_reward
#                     model.save(f"{model_dir}/{model.__class__.__name__}_best")
#
#                 # Log training results
#                 if log_dir is not None:
#                     with open(f"{log_dir}/{model.__class__.__name__}_training.txt", "a") as f:
#                         f.write(f"{timesteps_so_far},{mean_reward}\n")
#
#         # Training update
#         if isinstance(model, SAC):
#             if timesteps_so_far >= model.learning_starts:
#                 # Update progress for learning rate schedule
#                 model._current_progress_remaining = 1.0 - float(timesteps_so_far) / float(total_timesteps)
#                 model.train(gradient_steps=1)
#
#     pbar.close()
#     return model, list(recent_rewards)
#
#
# def train_agents_custom(total_timesteps: int = 1000000):
#     # Create timestamped directory for this training run
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_dir = f"training_runs/{timestamp}"
#     log_dir = f"{base_dir}/logs"
#     model_dir = f"{base_dir}/models"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
#
#     # Create environments
#     env = DummyVecEnv([make_env(0)])
#     env = VecNormalize(env, norm_obs=True, norm_reward=True)
#
#     # Create eval environment
#     eval_env = DummyVecEnv([make_env(1)])
#     eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
#
#     # Save the VecNormalize statistics
#     env_stats_path = os.path.join(model_dir, "vec_normalize.pkl")
#     env.save(env_stats_path)
#
#     models = {}
#     # Initialize and train each agent
#     for model_name, model_class, model_kwargs in [
#         ("SAC", SAC, {
#                         "learning_rate": 1e-4,
#                         "buffer_size": 1000000,
#                         "batch_size": 256,
#                         "tau": 0.005,
#                         "gamma": 0.99,
#                         "train_freq": 1,
#                         "gradient_steps": 1,
#                     }),
#         # ("PPO", PPO, {
#         #     "learning_rate": 2e-3,
#         #     "n_steps": 2048,
#         #     "batch_size": 64,
#         #     "n_epochs": 10,
#         #     "gamma": 0.99,
#         #     "gae_lambda": 0.95,
#         #     "clip_range": 0.2,
#         # }),
#         # ... (SAC and DDPG configurations remain the same)
#     ]:
#         print(f"\nTraining {model_name}...")
#
#         # Initialize model
#         model = model_class(
#             "MlpPolicy",
#             env,
#             verbose=0,
#             tensorboard_log=log_dir,
#             device="cuda" if torch.cuda.is_available() else "cpu",
#             **model_kwargs
#         )
#
#         # Train using custom loop
#         # model, rewards, lengths = custom_training_loop(
#         #     env=env,
#         #     model=model,
#         #     total_timesteps=total_timesteps,
#         #     batch_size=2048,
#         #     eval_freq=500,
#         #     eval_env=eval_env,
#         #     model_dir=model_dir,
#         #     log_dir=log_dir
#         # )
#         model, recent_rewards = custom_training_loop(
#             env=env,
#             model=model,
#             total_timesteps=1000000,
#             reward_window=10000,  # Keep track of last 1000 episode rewards
#             model_dir=model_dir,
#             log_dir=log_dir
#         )
#
#         # Save final model
#         final_model_path = f"{model_dir}/{model_name}_final"
#         model.save(final_model_path)
#
#         # Store model for evaluation
#         models[model_name] = model
#
#     return models, base_dir


def train_agents(total_timesteps: int = 1000000):
    # Create timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"training_runs/{timestamp}"
    log_dir = f"{base_dir}/logs"
    model_dir = f"{base_dir}/models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create vectorized environment
    #env = DummyVecEnv([make_env(0)])
    # env = DummyVecEnv([make_env(0)])
    #
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    #
    # # Save the VecNormalize statistics
    # env_stats_path = os.path.join(model_dir, "vec_normalize.pkl")
    # env.save(env_stats_path)

    # Create eval environment
    eval_env = DummyVecEnv([make_env(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    models = {}
    # Initialize and train each agent
    for model_name, model_class, model_kwargs in [

        ("SAC", SAC, {
            "learning_rate": 5e-4,
            "buffer_size": 1000000,
            "batch_size": 128 ,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            #'ent_coef': 'auto_0.4',
            #"use_sde": True,
            #"learning_starts": 10000,

        }),
        ("DDPG", DDPG, {
            "learning_rate": 1e-3,
            "buffer_size": 1000000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "episode"),
            "action_noise": NormalActionNoise(
                mean=np.zeros(eval_env.action_space.shape[0]),
                sigma=0.1 * np.ones(eval_env.action_space.shape[0])
            ),


        }),
        ("PPO", PPO, {
            "learning_rate": 2e-3,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        }),

    ]:
        print(f"\nTraining {model_name}...")

        # Create progress bar
        pbar = tqdm(total=total_timesteps, desc=f"Training {model_name}")

        # Create evaluation callback with progress bar
        eval_callback = TqdmCallback(
            pbar = pbar,
            eval_env=eval_env,
            best_model_save_path=f"{model_dir}/{model_name}_best",
            log_path=f"{log_dir}/{model_name}",
            eval_freq=10000,
            deterministic=False,
            render=False
        )

        # Initialize model
        model = model_class(
            "MlpPolicy",
            eval_env,
            verbose=0,  # Set to 0 to avoid competing with tqdm
            tensorboard_log=log_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **model_kwargs
        )

        # Train model
        print(1)
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

        # Save final model and training info
        final_model_path = f"{model_dir}/{model_name}_final"
        model.save(final_model_path)

        # Store model for evaluation
        models[model_name] = model

        pbar.close()

    return models, base_dir


def evaluate_agent(model, env, num_episodes=10):
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def load_and_test_model(model_dir: str, model_name: str, num_episodes: int = 10):
    """
    Load a saved model and test it in the environment
    """
    # Create environment
    env = DummyVecEnv([make_env(0)])

    # Load VecNormalize statistics
    env = VecNormalize.load(os.path.join(model_dir, "vec_normalize.pkl"), env)

    # Don't update normalization statistics during testing
    env.training = False
    env.norm_reward = False

    # Load the model
    if model_name == "PPO":
        model = PPO.load(os.path.join(model_dir, f"{model_name}_final"), env=env)
    elif model_name == "SAC":
        model = SAC.load(os.path.join(model_dir, f"{model_name}_final"), env=env)
    elif model_name == "DDPG":
        model = DDPG.load(os.path.join(model_dir, f"{model_name}_final"), env=env)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Evaluate the loaded model
    mean_reward, std_reward = evaluate_agent(model, env, num_episodes)
    print(f"Loaded {model_name} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    return model, env


if __name__ == "__main__":
    # Train the agents
    models, base_dir = train_agents(total_timesteps=100000)

    # print("\nEvaluating trained models...")
    # # Create evaluation environment
    # eval_env = DummyVecEnv([make_env(0)])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    # eval_env.training = False
    # eval_env.norm_reward = False
    #
    # # Evaluate each trained model
    # for name, model in models.items():
    #     mean_reward, std_reward = evaluate_agent(model, eval_env)
    #     print(f"{name} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    base_dir = 'training_runs/20250105_063354'
    print("\nTesting loaded models...")
    # Example of loading and testing a saved model
    for model_name in [ "SAC"]:
        loaded_model, _ = load_and_test_model(
            model_dir=f"{base_dir}/models",
            model_name=model_name,
            num_episodes=10
        )