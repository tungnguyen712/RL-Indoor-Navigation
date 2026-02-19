import os
import sys
import json
import time
import numpy as np
import pybullet as p

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from envs.indoor_maze_env import IndoorMazeEnv


class ReactiveLidarHeuristic:
    def __init__(self, 
                 obstacle_distance_m=0.3,
                 lidar_max_range=3.0,
                 turn_duration_steps=35,
                 turn_speed=0.8,
                 forward_speed=0.8,
                 angle_tolerance=0.2):
        # Convert physical distance to normalized lidar reading
        self.obstacle_threshold = obstacle_distance_m / lidar_max_range  # 0.3m / 3.0m = 0.1
        self.turn_duration_steps = turn_duration_steps
        self.turn_speed = turn_speed
        self.forward_speed = forward_speed
        self.angle_tolerance = angle_tolerance
        
        # State tracking for turn maneuver
        self.turning = False
        self.turn_steps_remaining = 0
    
    def predict(self, observation, deterministic=True):
        # Parse observation
        lidar = observation[:64]  # 64 lidar readings (normalized 0-1)
        goal_dist = observation[64]  # Normalized distance to goal
        goal_angle = observation[65]  # Relative angle to goal (normalized -1 to 1)
        
        action = np.zeros(2, dtype=np.float32)
        
        # Check forward-facing lidar rays only (center region, front 60 degrees)
        forward_lidar = lidar[26:38]
        min_forward_lidar = np.min(forward_lidar)
        
        # check if we should start a turn
        if not self.turning and min_forward_lidar < self.obstacle_threshold:
            # 90-degree right turn
            self.turning = True
            self.turn_steps_remaining = self.turn_duration_steps
        
        # Execute turn if in turning state
        if self.turning:
            # 90-degree right turn: left wheel forward, right wheel backward
            action[0] = self.turn_speed      # left wheel
            action[1] = -self.turn_speed     # right wheel
            self.turn_steps_remaining -= 1
            
            # Exit turning state when done
            if self.turn_steps_remaining <= 0:
                self.turning = False
        
        # Normal navigation toward goal
        else:
            # Goal angle is normalized to [-1, 1] representing [-π, π]
            if abs(goal_angle) < self.angle_tolerance:
                # Aligned with goal, move forward
                action[0] = self.forward_speed
                action[1] = self.forward_speed
            
            elif goal_angle > 0:
                # Goal is to the left, turn left while moving
                turn_strength = min(abs(goal_angle), 1.0)
                action[0] = self.forward_speed * (1.0 - turn_strength * 0.5)
                action[1] = self.forward_speed
            
            else:
                # Goal is to the right, turn right while moving
                turn_strength = min(abs(goal_angle), 1.0)
                action[0] = self.forward_speed
                action[1] = self.forward_speed * (1.0 - turn_strength * 0.5)
        
        return action, None


def load_mazes(folder_path):
    maze_layouts = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(".json"):
            path = os.path.join(folder_path, f)
            with open(path, "r") as fp:
                maze_layouts.append(json.load(fp))
    return maze_layouts


def evaluate_heuristic(policy, env, num_episodes=10, render=True, verbose=True):
    successes = 0
    failures = 0
    timeouts = 0
    episode_lengths = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        # Reset policy state for new episode
        policy.turning = False
        policy.turn_steps_remaining = 0
        
        obs, _ = env.reset(seed=ep)
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {ep + 1}/{num_episodes}")
            print(f"{'='*60}")
        
        while not (done or truncated):
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Print progress every 100 steps with action info
            if verbose and step_count % 100 == 0:
                lidar = obs[:64]
                forward_lidar = lidar[26:38]
                min_forward = np.min(forward_lidar)
                goal_angle = obs[65]
                state = "TURNING" if policy.turning else "NAVIGATING"
                if policy.turning:
                    print(f"Step {step_count}: {state} (steps left: {policy.turn_steps_remaining}) | fwd_lidar={min_forward:.3f} | goal_angle={goal_angle:.3f}")
                else:
                    print(f"Step {step_count}: {state} | fwd_lidar={min_forward:.3f} | goal_angle={goal_angle:.3f} | action=[{action[0]:.2f}, {action[1]:.2f}]")
            
            if render:
                time.sleep(1.0 / 60.0)
        
        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)
        
        # Determine outcome
        if info.get("is_success", False):
            successes += 1
            outcome = "SUCCESS"
        elif truncated:
            timeouts += 1
            outcome = "TIMEOUT"
        else:
            failures += 1
            outcome = "FAILURE"
        
        if verbose:
            print(f"Outcome: {outcome}")
            print(f"Steps: {step_count}")
            print(f"Total Reward: {total_reward:.2f}")
    
    # Compute statistics
    success_rate = successes / num_episodes
    failure_rate = failures / num_episodes
    timeout_rate = timeouts / num_episodes
    avg_episode_length = np.mean(episode_lengths)
    avg_episode_reward = np.mean(episode_rewards)
    
    results = {
        "num_episodes": num_episodes,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "timeout_rate": timeout_rate,
        "avg_episode_length": avg_episode_length,
        "avg_episode_reward": avg_episode_reward,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }
    
    return results


def print_results(results):
    print("\n" + "="*60)
    print("RESULTS FOR REACTIVE LIDAR HEURISTIC")
    print("="*60)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Successes: {results['successes']} ({results['success_rate']*100:.1f}%)")
    print(f"Failures: {results['failures']} ({results['failure_rate']*100:.1f}%)")
    print(f"Timeouts: {results['timeouts']} ({results['timeout_rate']*100:.1f}%)")
    print(f"Avg Steps: {results['avg_episode_length']:.1f}")
    print(f"Avg Reward: {results['avg_episode_reward']:.2f}")
    print("="*60)


def main():
    # Configuration
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_c_hard")
    num_episodes = 10
    render_mode = "gui"
    
    # Load evaluation mazes
    print(f"Loading mazes from: {eval_maze_dir}")
    eval_mazes = load_mazes(eval_maze_dir)
    print(f"Loaded {len(eval_mazes)} maze(s)")
    
    # Create environment
    env = IndoorMazeEnv(
        maze_layouts=eval_mazes,
        render_mode=render_mode,
        terminate_on_collision=True
    )
    
    if render_mode == "gui":
        # Hide GUI overlays for cleaner visualization
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Create policy with tuned parameters
    policy = ReactiveLidarHeuristic(
        obstacle_distance_m=0.3, # React when 0.3m from wall
        lidar_max_range=3.0,
        turn_duration_steps=35, # Steps for 90-degree turn
        turn_speed=0.8,
        forward_speed=0.8, 
        angle_tolerance=0.2
    )
    
    # Evaluate the policy
    results = evaluate_heuristic(
        policy=policy,
        env=env,
        num_episodes=num_episodes,
        render=(render_mode == "gui"),
        verbose=True
    )
    print_results(results)
    env.close()


if __name__ == "__main__":
    main()