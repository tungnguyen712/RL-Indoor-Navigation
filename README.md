# RL-based Indoor Navigation Robot Simulation
### Overview
This project implements a learning-based indoor navigation system using PyBullet and PPO. A mobile robot must navigate from a random start to a random goal inside an indoor maze-like environment, using only local lidar observations.

### Sensors
1. 2d Lidar (ray casting)
2. Contact sensors (collision detection for penalty)

### Observation space
[64 lidar distances (normalized), goal_distance (normalized), goal_angle (normalized)]

### Task
At each episode, the robot will spawn at a random valid location. The goal position is randomly sampled such that there is a valid path between the starting and goal point. The starting point and goal point are guaranteed to not spawn too close together. The robot will attempt to navigate toward the goal while avoiding walls.

### Reward
**Terminal Rewards:**
- +150 for reaching the goal within 0.8m radius
- -20 for collision with walls

**Step-based Rewards:**
- Progress reward: +10.0 × (distance improvement)
- Step penalty: -0.1 per step - encourages efficiency (350 steps = -35 total)
- Standing still penalty: -2.0 per step when speed < 0.05 m/s

**Proximity Bonus (when within 1.5m of goal):**
- +4.0 × (proximity) if robot is moving (speed > 0.05 m/s) OR making progress (> 0.005m improvement)
- -1.5 if hovering near goal without movement - prevents exploitation

### Setup and Installation
1. **Clone the repository:**
```bash
git clone <repository-url>
cd rl-based-indoor-navigation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

**Train the model:**
```bash
python -m training.train_ppo
```
Training runs for 8M timesteps with 8 parallel environments (~2-3 hours on CPU).

**Evaluate the trained model:**
```bash
python -m training.evaluate
```
Evaluates the best model on 20 episodes and saves videos of successful runs to `/videos`.