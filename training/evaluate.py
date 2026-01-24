import os
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
import time
import cv2

from envs.indoor_maze_env import IndoorMazeEnv
from training.train_ppo import load_mazes

EPISODES = 20
MAX_STEPS_PER_EPISODE = 1000

def main():
    successes = 0
    collisions = 0
    timeouts = 0
    ep_rewards = []
    ep_steps = []
    final_dists = []

    # get maze and model paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    eval_maze_dir = os.path.join(project_root, "assets", "eval")
    model_path = os.path.join(project_root, "models", "best_model.zip")
    videos_dir = os.path.join(project_root,"videos", "7x7_maze")

    # load mazes and model
    eval_mazes = load_mazes(eval_maze_dir)
    model = PPO.load(model_path)

    env = IndoorMazeEnv(maze_layouts=eval_mazes, render_mode="gui")

    for ep in range(EPISODES):
        print(f"\nEpisode {ep+1}/{EPISODES}:")
        obs, _ = env.reset(seed=ep)
        print(f"Start: ({env.start_pos[0]:.2f}, {env.start_pos[1]:.2f})")
        print(f"Goal:  ({env.goal_pos[0]:.2f}, {env.goal_pos[1]:.2f})")
        start_dist = np.linalg.norm(np.array(env.start_pos) - np.array(env.goal_pos))
        print(f"Initial distance to goal: {start_dist:.2f}m")
        
        # prepare for frame capture
        video_path = os.path.join(videos_dir, f"episode_{ep+1:03d}.mp4")
        frames = []
        
        total_reward = 0.0
        terminal = False
        ep_start_time = time.time()

        for step in range(MAX_STEPS_PER_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            
            # capture frame
            width, height, rgb, depth, seg = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]  # remove alpha
            frames.append(frame)
            
            time.sleep(0.03)

            # check actual robot speed
            lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
            speed = (lin_vel[0]**2 + lin_vel[1]**2) ** 0.5
            # print("Speed:", speed) # best in [0.5, 1.5]

            # terminal state
            if terminated or truncated:
                # get final distance to goal
                pos, orie = p.getBasePositionAndOrientation(env.robot_id)
                robot_xy = np.array([pos[0], pos[1]], dtype=np.float32)
                goal_xy = np.array(env.goal_pos, dtype=np.float32)
                dist = float(np.linalg.norm(robot_xy - goal_xy))

                outcome = ""
                if env.check_success():
                    successes += 1
                    outcome = "SUCCESS"
                elif env.check_collision():
                    collisions += 1
                    outcome = "COLLISION"
                else:
                    timeouts += 1
                    outcome = "TIMEOUT"
                    
                ep_time = time.time() - ep_start_time
                print(f"{outcome} after {step + 1} steps ({ep_time:.1f}s)")
                print(f"Final distance: {dist:.2f}m | Reward: {total_reward:.2f}")
                
                # save video only if successful
                if outcome == "SUCCESS" and len(frames) > 0:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
                    for frame in frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    print(f"Video saved: {video_path}")
                
                ep_rewards.append(total_reward)
                ep_steps.append(step + 1)
                final_dists.append(dist)
                terminal = True
                break
        
        # handle edge case where max steps reached but not terminal
        if not terminal:
            pos, _ = p.getBasePositionAndOrientation(env.robot_id)
            
            robot_xy = np.array([pos[0], pos[1]], dtype=np.float32)
            goal_xy = np.array(env.goal_pos, dtype=np.float32)
            dist = float(np.linalg.norm(robot_xy - goal_xy))

            timeouts += 1
            ep_time = time.time() - ep_start_time
            print(f"Timeout after {MAX_STEPS_PER_EPISODE} steps ({ep_time:.1f}s)")
            print(f"Final distance: {dist:.2f}m | Reward: {total_reward:.2f}")
            
            ep_rewards.append(total_reward)
            ep_steps.append(MAX_STEPS_PER_EPISODE)
            final_dists.append(dist)
        
    env.close()
    if len(ep_rewards) > 0 and len(final_dists) > 0 and len(ep_steps) > 0:
        print("Success rate:", successes / EPISODES)
        print("Collision rate:", collisions / EPISODES)
        print("Timeout rate:", timeouts / EPISODES)
        print("Average reward:", float(np.mean(ep_rewards)))
        print("Average steps:", float(np.mean(ep_steps)))
        print("Average final distance:", float(np.mean(final_dists)))

if __name__ == "__main__":
    main()