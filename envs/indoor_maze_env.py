import time
import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data

import json
import os
class IndoorMazeEnv(gym.Env):
    metadata = {
        "render_modes": ["direct", "gui"],
        "render_fps": 60,
    }

    def __init__(self, maze_layouts, render_mode="direct"):
        super().__init__()
        self.maze_layouts = maze_layouts
        self.grid = None
        self.wall_ids = []
        self.physics_steps_per_action = 20
        
        self.wall_height = 0.4
        self.lidar_num_rays = 64
        self.lidar_fov = 180
        self.lidar_max_range = 3.0
        self.lidar_debug_line_ids = []
        self.debug_lidar_every = 10

        self.max_episode_steps = 350
        self.goal_radius = 0.8  # increased from 0.5 for better success rate
        self.goal_marker_radius = 0.8

        # action = [wheel_angular_speed, steering_angle]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        # [LIDAR distances + goal direction + goal angle]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * self.lidar_num_rays + [0.0, -1.0], dtype=np.float32),
            high=np.array([1.0] * self.lidar_num_rays + [1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # reward params
        self.goal_reward = 150.0 # increased from 100 to significantly reward success
        self.collision_penalty = 20.0
        self.dist_weight = 10.0  # increased from 8.0 to reward progress > standing still
        self.step_penalty = 0.1  # encourage efficiency
        self.proximity_bonus_threshold = 1.5
        self.proximity_bonus_weight = 4.0  # increased from 3.0 for stronger goal attraction
        self.standing_still_penalty = 2.0  # penalty for not moving
        self.min_speed_threshold = 0.05  # minimum speed to be considered moving

        # robot paremeters
        self.max_wheel_rad_per_sec = 40.0  # increased from 8.0 for faster movement
        self.max_steering_angle = 0.5  # increased from 0.35 for sharper turns

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        if p.isConnected():
            p.disconnect()

        if self.render_mode == "gui":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # basic pybullet setup
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # load robot
        self.robot_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1])

        # create goal marker
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.goal_marker_radius)
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.goal_marker_radius,
            rgbaColor=[0.0, 1.0, 0.0, 0.2],
        )

        self.goal_marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, self.goal_marker_radius],
        )
        
        # add strong green outline for visibility
        p.changeVisualShape(self.goal_marker_id, -1, rgbaColor=[0.0, 1.0, 0.0, 0.2])
        p.changeVisualShape(self.goal_marker_id, -1, specularColor=[0.0, 1.0, 0.0])

        # simplify GUI and set up camera
        if self.render_mode == "gui":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            # set initial camera position on birdseye view
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[2.0, 2.0, 0]
            )

        # get joint ids
        self.rear_wheel_joints = []
        self.front_wheel_joints = []
        self.steering_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            if joint_name in ["left_rear_wheel_joint", "right_rear_wheel_joint"]:
                self.rear_wheel_joints.append(i)
            elif joint_name in ["left_front_wheel_joint", "right_front_wheel_joint"]:
                self.front_wheel_joints.append(i)
            elif joint_name in ["left_steering_hinge_joint", "right_steering_hinge_joint"]:
                self.steering_joints.append(i)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_goal_dist = None

        # pick a maze layout
        maze = self.np_random.choice(self.maze_layouts)
        self.grid = maze["grid"]
        self.cell_size = maze.get("cell_size", 0.5)

        # reset walls
        self.remove_walls()
        self.create_walls()

        # get start + goal positions
        self.start_pos, self.goal_pos = self.sample_random_position()

        # reset robot position, orientation, and velocity
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [self.start_pos[0], self.start_pos[1], 0.1],
            p.getQuaternionFromEuler([0, 0, self.np_random.uniform(-np.pi, np.pi)]),
        )
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # stop all motors
        for i in self.rear_wheel_joints:
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

        for i in self.steering_joints:
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=0
            )
        
        for _ in range(10):
            p.stepSimulation()
        
        # verify no immediate collision at spawn
        if self.check_collision():
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.start_pos[0], self.start_pos[1], 0.15],  # slightly higher
                p.getQuaternionFromEuler([0, 0, self.np_random.uniform(-np.pi, np.pi)]),
            )
            for _ in range(10):
                p.stepSimulation()

        # reset goal marker position
        p.resetBasePositionAndOrientation(
            self.goal_marker_id,
            [self.goal_pos[0], self.goal_pos[1], self.goal_marker_radius],
            [0, 0, 0, 1],
        )

        # reset lidar debug lines
        if self.render_mode == "gui":
            for line_id in self.lidar_debug_line_ids:
                p.removeUserDebugItem(line_id)
            self.lidar_debug_line_ids = []
            
            # Update camera to focus on the current maze
            maze_center_x = (len(self.grid[0]) * self.cell_size) / 2
            maze_center_y = (len(self.grid) * self.cell_size) / 2
            p.resetDebugVisualizerCamera(
                cameraDistance=max(len(self.grid), len(self.grid[0])) * self.cell_size * 0.8,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[maze_center_x, maze_center_y, 0]
            )

        obs = self.get_observation()

        info = {}

        return obs, info

    def step(self, action):
        self.control_robot(action)
        for _ in range(self.physics_steps_per_action):
            p.stepSimulation()

        obs = self.get_observation()
        reward = self.compute_reward()
        self.step_count += 1
        terminated = self.check_success() or self.check_collision()
        truncated = self.step_count >= self.max_episode_steps
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected():
            p.disconnect()

    def create_walls(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 1:
                    wall_width = self.cell_size * 0.98
                    wall_depth = self.cell_size * 0.98
                    # create wall collision and visual shapes
                    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2])
                    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

                    # create wall body
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape_id,
                        baseVisualShapeIndex=visual_shape_id,
                        basePosition=[
                            col * self.cell_size + self.cell_size / 2,
                            row * self.cell_size + self.cell_size / 2,
                            self.wall_height / 2,
                        ]
                    )
                    self.wall_ids.append(body_id)

    def remove_walls(self):
        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []

    def sample_random_position(self, min_dist=1, max_tries=100):
        free_cells = [(row, col) for row in range(len(self.grid)) for col in range(len(self.grid[0])) if self.grid[row][col] == 0]
        start_cell = self.np_random.choice(free_cells)
        
        start_pos = (
            start_cell[1] * self.cell_size + self.cell_size / 2,
            start_cell[0] * self.cell_size + self.cell_size / 2,
        )

        for _ in range(max_tries):
            goal_cell = self.np_random.choice(free_cells)
            goal_pos = (
                goal_cell[1] * self.cell_size + self.cell_size / 2,
                goal_cell[0] * self.cell_size + self.cell_size / 2,
            )
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist >= min_dist:
                return start_pos, goal_pos
        
        while True:
            goal_cell = self.np_random.choice(free_cells)
            if tuple(goal_cell) != tuple(start_cell):
                goal_pos = (
                    goal_cell[1] * self.cell_size + self.cell_size / 2,
                    goal_cell[0] * self.cell_size + self.cell_size / 2,
                )
                return start_pos, goal_pos

    def control_robot(self, action):
        v_norm, w_norm = action
        target_v = v_norm * self.max_wheel_rad_per_sec
        target_w = w_norm * self.max_steering_angle

        # set wheel velocities
        for joint_index in self.rear_wheel_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_v,
                force=150,
            )

        # set steering angles
        for joint_index in self.steering_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_w,
                force=25,
            )

    def get_observation(self):
        # get robot position and yaw (heading)
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y, robot_z = position
        euler = p.getEulerFromQuaternion(orientation)
        yaw = euler[2]

        # generate lidar ray angles centered on robot yaw spanning lidar_fov
        angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.lidar_num_rays) *np.pi/180
        ray_from = []
        ray_to = []
        for a in angles:
            theta = yaw + a
            dx = np.cos(theta)
            dy = np.sin(theta)

            ray_from.append([robot_x, robot_y, robot_z + 0.1])
            ray_to.append([
                robot_x + dx * self.lidar_max_range,
                robot_y + dy * self.lidar_max_range,
                robot_z + 0.1
            ])

        # raycast each direction up to lidar_max_range, normalize distances to [0, 1]
        results = p.rayTestBatch(ray_from, ray_to)
        # reset lidar debug lines
        # if self.render_mode == "gui" and self.step_count % self.debug_lidar_every == 0:
        #     # remove previous lines
        #     for line_id in self.lidar_debug_line_ids:
        #         p.removeUserDebugItem(line_id)
        #     self.lidar_debug_line_ids = []
        lidar_distances = []

        for i, r in enumerate(results):
            hit_body = r[0]
            hit_fraction = r[2]

            # ignore self robot hits
            if hit_body == self.robot_id:
                hit_fraction = 1.0

            lidar_distances.append(hit_fraction)

            # show lidar debug lines
            # if self.render_mode == "gui" and self.step_count % self.debug_lidar_every == 0:
            #     # compute endpoint
            #     fx, fy, fz = ray_from[i]
            #     tx, ty, tz = ray_to[i]

            #     end = [
            #         fx + (tx - fx) * hit_fraction,
            #         fy + (ty - fy) * hit_fraction,
            #         fz + (tz - fz) * hit_fraction,
            #     ]
            #     line_id = p.addUserDebugLine(
            #         ray_from[i],
            #         end,
            #         lineColorRGB=[1, 0, 0],  # red
            #         lineWidth=1,
            #         lifeTime=0,  # persists until removed
            #     )
            #     self.lidar_debug_line_ids.append(line_id)
        norm_lidar = np.array(lidar_distances, dtype=np.float32)

        # compute goal distance and normalize to [0, 1]
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))
        max_dist = np.linalg.norm(np.array([len(self.grid)*self.cell_size, len(self.grid[0])*self.cell_size]))
        norm_goal_dist = goal_dist / max_dist
        norm_goal_dist = np.clip(norm_goal_dist, 0.0, 1.0)

        # compute goal angle relative to robot's heading, wrap to [-pi, pi], normalize to [-1, 1]
        goal_angle = np.arctan2(goal_dy, goal_dx)
        relative_angle = goal_angle - yaw
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
        norm_goal_angle = relative_angle / np.pi

        # concatenate lidar + [goal distance + goal angle] into 1 1D float32 array
        observation = np.concatenate([norm_lidar, np.array([norm_goal_dist, norm_goal_angle], dtype=np.float32)])
        return observation


    def compute_reward(self):
        # get robot position
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        
        # get velocity (speed) to discourage hovering
        velocity, _ = p.getBaseVelocity(self.robot_id)
        speed = np.linalg.norm(velocity[:2])  # only x,y velocity

        # get goal distance
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))

        reward = 0.0
        # progress based reward - heavily weighted for fast robot
        if self.prev_goal_dist is not None:
            dist_delta = self.prev_goal_dist - goal_dist
        else:
            dist_delta = 0.0
        reward += self.dist_weight * dist_delta
        self.prev_goal_dist = goal_dist
        
        # penalize standing still to prevent timeouts
        if speed < self.min_speed_threshold:
            reward -= self.standing_still_penalty
        
        # proximity bonus
        if goal_dist < self.proximity_bonus_threshold:
            # give proximity bonus if robot is moving or making progress
            if speed > self.min_speed_threshold or dist_delta > 0.005:
                proximity_bonus = (self.proximity_bonus_threshold - goal_dist) * self.proximity_bonus_weight
                reward += proximity_bonus
            else:
                # moderate penalty for hovering near goal without movement
                reward -= 1.5

        # terminal rewards/penalties
        if self.check_success():
            reward += self.goal_reward
        if self.check_collision():
            reward -= self.collision_penalty
        
        # step penalty
        reward -= self.step_penalty
        return reward

    def check_success(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        dist = np.linalg.norm(np.array([robot_x, robot_y]) - np.array(self.goal_pos))
        return dist <= self.goal_radius

    def check_collision(self):
        contacts = p.getContactPoints(bodyA=self.robot_id)
        wall_ids_set = set(self.wall_ids)
        for contact in contacts:
            bodyA_id = contact[1]
            bodyB_id = contact[2]
            if bodyA_id == self.robot_id:
                if bodyB_id in wall_ids_set:
                    return True
            else:
                if bodyA_id in wall_ids_set:
                    return True
        return False
    

def main():
    # load mazes from json files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    maze_dir = os.path.join(project_root, "assets", "train")

    # choose a small subset first for debugging
    maze_files = [
        "train_maze_01.json",
    ]

    maze_layouts = []
    for f in maze_files:
        path = os.path.join(maze_dir, f)
        with open(path, "r") as fp:
            maze_layouts.append(json.load(fp))

    # create env in GUI mode for visual debugging
    env = IndoorMazeEnv(maze_layouts=maze_layouts, render_mode="gui")

    try:
        # repeated reset test
        for ep in range(10):
            obs, info = env.reset(seed=ep)

            print(f"\nEpisode {ep} reset")
            print("obs shape:", obs.shape, "dtype:", obs.dtype)
            print("num walls:", len(env.wall_ids))
            print("start:", env.start_pos, "goal:", env.goal_pos)

            # random actions
            for t in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if t % 20 == 0:
                    pos, _ = p.getBasePositionAndOrientation(env.robot_id)
                    print("Reward:", reward)
                    print("Terminated:", terminated, "\tTruncated:", truncated)
                    print("Robot pos: ", pos[0], pos[1])
                    print("Goal dist (normed)", obs[env.lidar_num_rays])

                if terminated or truncated:
                    print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated})")
                    break
                time.sleep(1.0 / 60.0)
    finally:
        env.close()

if __name__ == "__main__":
    main()
