import os
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from envs.indoor_maze_env import IndoorMazeEnv

NUM_ENVS = 8  # more parallel environments for diverse experiences
POLICY = "MlpPolicy"
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.03  # increased from 0.02
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 8_000_000  # increased from 4M
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 10  # increased from 5 for more robust evaluation

def load_mazes(folder_path):
    maze_layouts = []
    for f in os.listdir(folder_path):
        if f.endswith(".json"):
            path = os.path.join(folder_path, f)
            with open(path, "r") as fp:
                maze_layouts.append(json.load(fp))
    return maze_layouts

def make_env(maze_layouts, render_mode="direct"):
    def _init():
        env = IndoorMazeEnv(maze_layouts=maze_layouts, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

def main():
    # get maze paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_maze_dir = os.path.join(project_root, "assets", "train")
    eval_maze_dir = os.path.join(project_root, "assets", "eval")
    models_dir = os.path.join(project_root, "models")
    logs_dir = os.path.join(project_root, "logs")

    # load mazes
    train_mazes = load_mazes(train_maze_dir)
    eval_mazes = load_mazes(eval_maze_dir)

    # better than gpu for this task
    device = "cpu"

    # create parallel environments for faster training
    train_env = SubprocVecEnv([make_env(train_mazes, render_mode="direct") for _ in range(NUM_ENVS)])
    eval_env = DummyVecEnv([make_env(eval_mazes, render_mode="direct")])

    model = PPO(
        POLICY,
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY_COEF,
        vf_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=logs_dir,
        device=device
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    model.save(os.path.join(models_dir, "ppo_maze"))

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
