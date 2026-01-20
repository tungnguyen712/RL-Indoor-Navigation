# RL-based Indoor Navigation Robot Simulation
### Overview
This project implements a learning-based indoor navigation system using PyBullet and PPO. A mobile robot must navigate from a random start to a random goal inside an indoor maze-like environment, using only local lidar observations.

### Sensors
1. 2d Lidar (ray casting)
2. Contact sensors (collision detection for penalty)

### Observation space
[lidar_distance, goal_distance, goal_angle, linear_velocity, angular_velocity]

### Task
At each episode, the robot will spawn at a random valid location. The goal position is randomly sampled such that there is a valid path between the starting and goal point. The starting point and goal point are guaranteed to not spawn too close together. The robot will attempt to navigate toward the goal while avoiding walls.

### Rewards
1. -distance to goal
2. +10 for reaching goal
3. -10 for collision
4. -0.01 per timestep