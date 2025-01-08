from gymnasium import Env
from gymnasium import spaces
import random
import numpy as np
import pybullet as p
import pybullet_data
from env import AutonomousCarEnv


class AutonomousCarGymnasiumEnv(Env):
    metadata = {'render_modes': ['human']}
    def __init__(self):
        super().__init__()
        #self.client = p.connect(p.DIRECT)
        #self.car_env = AutonomousCarEnv()
        self.car_env = AutonomousCarEnv()
        #self.car_env = AutonomousCarEnv(client=self.client)

        # Define action space (4 wheel velocities)
        self.action_space = spaces.Box(
            low=np.array([-1, 0]),
            high=np.array([1, 1.0]),
            dtype=np.float32
        )
        self.total_reward = 0
        self.total_time_steps = 0



        # Define observation space (only LiDAR readings)
        self.observation_space = spaces.Box(
            low=0.0,  # Minimum distance reading
            high=4.0,  # Maximum LiDAR range
            shape=(128,),  # 10 LiDAR rays
            dtype=np.float32
        )

        # Constants for action scaling
        #self.max_steering_angle = 0.5  # ±0.5 radians (about ±30 degrees)
        self.max_steering_angle = 1.3  # ±0.785 radians (±45 degrees) [π/4]
        self.max_speed = 20.0  # Maximum wheel velocity
        self.previous_distance_to_goal = self.car_env.road_length

        # Previous steering angle for smooth control
        self.prev_steering = 0.0

        # Add progress tracking variables
        self.steps_without_progress = 0
        self.max_steps_without_progress = 5000  # Adjust this value as needed
        self.progress_threshold = 0.01  # Minimum distance improvement needed
        self.distance_improvement_sum = 0
        #self.reached_15 = False

    def _convert_actions_to_control(self, action):
        """Convert normalized actions to actual control signals"""
        steering = action[0] * self.max_steering_angle
        speed = action[1] * self.max_speed

        # Smooth steering changes
        steering = 0.8 * self.prev_steering + 0.2 * steering
        self.prev_steering = steering

        # Calculate individual wheel velocities based on steering
        # Implement differential drive
        left_speed = speed - steering * speed * 20
        right_speed = speed + steering * speed * 20

        # Convert to four wheel velocities
        wheel_velocities = [
            left_speed,
            right_speed,  # Front right
            left_speed,  # Front left
            right_speed,  # Rear right
        ]

        return wheel_velocities

    def _process_lidar_data(self, lidar_data):
        """Convert raw LiDAR data to distances"""
        distances = np.array([
            ray[2] * self.car_env.ray_length
            for ray in lidar_data
        ], dtype=np.float32)

        return distances

    # def _compute_reward(self, dict_env, action):
    #     done = False
    #     reward = 0
    #
    #     # Get current position and calculate distances
    #     car_pos = np.array(dict_env['car_position'])
    #     goal_pos = np.array(self.car_env.end_point)
    #     distance_to_goal = np.linalg.norm(car_pos - goal_pos)
    #
    #     # 1. Progress Reward - Smoother gradient
    #     distance_improvement = self.previous_distance_to_goal - distance_to_goal
    #     progress_reward = distance_improvement * 10  # Smaller scale to encourage exploration
    #     reward += progress_reward
    #
    #     # 2. Distance-based reward - Gentler curve
    #     distance_reward = -1 * distance_to_goal  # Linear penalty instead of cubic
    #     reward += distance_reward
    #
    #     # 3. Milestone Rewards - Smaller and more frequent
    #     if distance_to_goal <= 23 and self.reached_18 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_18 = True
    #         print("Reached 23m")
    #         reward += 10000
    #     elif self.reached_18:
    #         pass
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -30.0 * (distance_to_goal - 23)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 20 and self.reached_15 is False:
    #         # Reward for reaching goal
    #         self.reached_15 = True
    #         print("Reached 20m")
    #         reward += 500
    #     elif self.reached_15:
    #         pass
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -10.0 * (distance_to_goal - 20)
    #         reward += remaining_distance_penalty
    #
    #     if distance_to_goal <= 18 and self.reached_12 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_12 = True
    #         print("Reached 18m")
    #         reward += 500
    #     elif self.reached_12:
    #         pass
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -9.0 * (distance_to_goal - 17)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 17 and self.reached_6 is False:
    #         # Reward for reaching goal
    #         self.reached_6 = True
    #         print("Reached 17m")
    #         reward += 500
    #     elif self.reached_6:
    #         pass
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -7.0 * (distance_to_goal - 14)
    #         reward += remaining_distance_penalty
    #
    #     # 4. Exploration Bonus
    #     # Add entropy bonus based on action
    #     entropy_bonus = -0.1 * (action[0] ** 2 + action[1] ** 2)  # Encourage varied actions
    #     reward += entropy_bonus
    #
    #     # 5. Collision Handling - Less severe
    #     if dict_env['collision']['detected']:
    #         collision_penalty = -100  # Reduced penalty
    #         reward += collision_penalty
    #         print(f'Collision detected! Direction: {dict_env["collision"]["direction"]}, '
    #               f'Total_reward: {reward}m {dict_env["collision"]["id"]:} Total_timesteps: {self.total_time_steps}')
    #         done = True
    #         return reward, done
    #
    #     # 6. Anti-stalling with exploration encouragement
    #     if self.steps_without_progress >= self.max_steps_without_progress:
    #         stall_penalty = -50  # Reduced penalty
    #         #exploration_bonus = 20 * np.random.random()  # Random bonus to encourage exploration
    #         reward += stall_penalty #+ exploration_bonus
    #         print(f"Stuck at distance: {distance_to_goal:.2f}, Total steps: {self.total_time_steps}")
    #         done = True
    #         return reward, done
    #
    #     # 7. Goal Achievement - More reasonable reward
    #     if distance_to_goal < 3.0:
    #         goal_reward = 5000  # Reduced from 10000
    #         reward += goal_reward
    #         print(f'Goal reached! Total steps: {self.total_time_steps}, Final reward: {reward:.2f}')
    #         done = True
    #         return reward, done
    #
    #     # Update previous distance for next iteration
    #     self.previous_distance_to_goal = distance_to_goal
    #
    #     return reward, done

    # def _compute_reward(self, dict_env, action):
    #     done = False
    #     reward = 0
    #
    #     # Get current position and calculate distances
    #     car_pos = np.array(dict_env['car_position'])
    #     goal_pos = np.array(self.car_env.end_point)
    #     distance_to_goal = np.linalg.norm(car_pos - goal_pos)
    #
    #     # 1. Progress Shaping
    #     # Use difference in potential functions as shaping reward
    #     # Potential = -distance_to_goal^2 (negative because closer is better)
    #     current_potential = -(distance_to_goal ** 2)
    #     previous_potential = -(self.previous_distance_to_goal ** 2)
    #     shaping_reward = 0.1 * (current_potential - previous_potential)  # Scale factor to balance with other rewards
    #     reward += shaping_reward
    #     #print(self.reached_15)
    #
    #     if distance_to_goal <= 23 and self.reached_18 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_18 = True
    #         print("Reached 23m")
    #         reward += 10000
    #     elif self.reached_18:
    #         pass
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -3000.0 * (distance_to_goal - 23)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 20 and self.reached_15 is False:
    #         # Reward for reaching goal
    #         self.reached_15 = True
    #         print("Reached 20m")
    #         reward += 10000
    #     elif self.reached_15:
    #         pass
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -1000.0 * (distance_to_goal - 20)
    #         reward += remaining_distance_penalty
    #
    #     if distance_to_goal <= 17 and self.reached_12 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_12 = True
    #         print("Reached 17m")
    #         reward += 10000
    #     elif self.reached_18:
    #         pass
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -900.0 * (distance_to_goal - 17)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 14 and self.reached_6 is False:
    #         # Reward for reaching goal
    #         self.reached_6 = True
    #         print("Reached 14m")
    #         reward += 10000
    #     elif self.reached_6:
    #         pass
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -700.0 * (distance_to_goal - 14)
    #         reward += remaining_distance_penalty
    #
    #
    #
    #
    #     # 2. Distance-based Component
    #     # Smooth gradient based on distance to goal
    #     distance_reward = -( distance_to_goal ** 3)/2  # Linear penalty for distance
    #     reward += distance_reward
    #
    #     # 3. Action Smoothness
    #     # Penalize large steering changes for smoother driving
    #     steering_change = abs(action[0] - self.prev_steering)
    #     smoothness_penalty = -1.0 * steering_change
    #     reward += smoothness_penalty
    #
    #     # 4. Forward Progress Bonus
    #     # Reward moving towards goal more directly
    #     forward_progress = self.previous_distance_to_goal - distance_to_goal
    #     if forward_progress > 0:
    #         progress_bonus = 600 * forward_progress
    #         reward += progress_bonus
    #
    #     # 5. Collision Avoidance
    #     if dict_env['collision']['detected']:
    #         # Graduated collision penalty based on impact speed
    #         speed = abs(action[1])
    #         collision_penalty = -10000.0 * (1 + speed)
    #         reward += collision_penalty
    #
    #         print(f'Collision detected! Direction: {dict_env["collision"]["direction"]}, '
    #               f'Total_reward: {reward}m {dict_env["collision"]["id"]:} Total_timesteps: {self.total_time_steps }')
    #         done = True
    #         return reward, done
    #
    #     # 6. Proximity Warning
    #     # Soft penalty for getting too close to obstacles
    #     # if dict_env['collision']['min_distance'] < 0.5:
    #     #     proximity_penalty = -10.0 * (0.5 - dict_env['collision']['min_distance'])
    #     #     reward += proximity_penalty
    #
    #     # 7. Goal Achievement
    #     if distance_to_goal < 3.0:
    #         # Bonus for reaching goal, scaled by efficiency
    #         steps_factor = max(0, 1 - (self.total_time_steps / 1000))  # Assume 1000 steps is baseline
    #         goal_reward = 10000.0 * (1 + steps_factor)
    #         reward += goal_reward
    #         print(f'Goal reached! Total steps: {self.total_time_steps}, Final reward: {reward:.2f}')
    #         done = True
    #         return reward, done
    #
    #     distance_improvement = self.previous_distance_to_goal - distance_to_goal
    #     if distance_improvement > self.progress_threshold:
    #         self.steps_without_progress = 0
    #         progress_bonus = 50  # Add explicit bonus for making progress
    #         self.previous_distance_to_goal = distance_to_goal
    #     else:
    #         self.steps_without_progress += 1
    #         progress_bonus = 0
    #
    #     # 8. Anti-stalling Component
    #     if self.steps_without_progress >= self.max_steps_without_progress:
    #         stall_penalty = -15000.0
    #         reward += stall_penalty
    #         #print(f"Stuck at distance: {distance_to_goal:.2f}, Total steps: {self.total_time_steps}, Total reward: {self.total_reward:.2f }")
    #         print(f"Stuck at distance: {distance_to_goal:.2f}, Total steps: {self.total_time_steps}, Total reward: {reward:.2f}")
    #
    #         done = True
    #         return reward, done
    #
    #     # Update previous distance for next iteration
    #     self.previous_distance_to_goal = distance_to_goal
    #
    #     # Track cumulative reward
    #     self.total_reward += reward
    #
    #     return reward, done

    # def _compute_reward(self, dict_env, action):
    #     done = False
    #     penalty_back = 0
    #     collision_reward = 0
    #
    #     distance_to_goal = np.linalg.norm(np.array(dict_env['car_position']) - np.array(self.car_env.end_point))
    #     distance_improvement = self.previous_distance_to_goal - distance_to_goal
    #
    #     # Stronger progress reward that scales with distance
    #     progress_reward = -(distance_to_goal ** 2) / 10  # Increased weight from /50 to /25
    #
    #     # Add time penalty to encourage faster completion
    #     #time_penalty = -1  # Small constant penalty per step
    #
    #     # Add movement penalty to discourage staying still
    #     #movement_magnitude = np.linalg.norm(action)
    #     #movement_reward = movement_magnitude * 2  # Reward for taking actions
    #
    #     if distance_to_goal > self.car_env.road_length:
    #         penalty_back = -(distance_to_goal ** 3)/10  # Increased penalty for going backwards
    #
    #     # Progress tracking
    #     if distance_improvement > self.progress_threshold:
    #         self.steps_without_progress = 0
    #         progress_bonus = 50  # Add explicit bonus for making progress
    #         self.previous_distance_to_goal = distance_to_goal
    #     else:
    #         self.steps_without_progress += 1
    #         progress_bonus = 0
    #
    #     # Getting stuck penalty
    #     if self.steps_without_progress >= self.max_steps_without_progress:
    #         # base_penalty = -(distance_to_goal ** 2)
    #         # distance_multiplier = np.exp(distance_to_goal / self.car_env.road_length)
    #         # progress_factor = (distance_to_goal / self.car_env.road_length) * 2  # Increased penalty
    #         #
    #         # reward = base_penalty * distance_multiplier * progress_factor
    #         reward = -10
    #         print(
    #             f"Stuck at distance: {distance_to_goal:.2f}, Total Reward: {self.total_reward:.2f}, Steps: {self.total_time_steps}, action: {action}")
    #         done = True
    #         return reward, done
    #
    #     # Goal reached reward
    #     if distance_to_goal < 3.0:
    #         # Make goal reward scale with steps taken to encourage efficiency
    #         time_bonus = max(0, 5000 - self.total_time_steps)  # Bonus decreases with more steps
    #         reward = 50000 + time_bonus
    #         print('Reached goal', self.total_reward, self.total_time_steps)
    #         done = True
    #         return reward, done
    #
    #     # Collision penalty
    #     if dict_env['collision']['detected']:
    #         print('collision detected', dict_env['collision']['detected'], dict_env['collision']['direction'],
    #               dict_env['collision']['min_distance'], dict_env['collision']['id'])
    #         collision_reward = -100  # Increased from -50 to make collisions more significant
    #
    #     total_reward = (
    #             progress_reward +
    #             collision_reward +
    #             penalty_back +
    #             progress_bonus
    #             #time_penalty +
    #             #movement_reward
    #     )
    #
    #     self.total_reward += total_reward
    #     return total_reward, done

    # def _compute_reward(self, dict_env, action):
    #     done = False
    #     reward = 0
    #     # print(dict_env)
    #     penalty_back = 0
    #     collision_reward = 0
    #
    #     distance_to_goal = np.linalg.norm(np.array(dict_env['car_position']) - np.array(self.car_env.end_point))
    #
    #     # Check if making progress towards goal
    #     distance_improvement =self.previous_distance_to_goal - distance_to_goal
    #     # if distance_to_goal < self.previous_distance_to_goal:
    #     #     reward += ((self.car_env.road_length - distance_to_goal) ** 3)/25  # + (self.car_env.road_length - distance_to_goal)/3
    #     # else:
    #     #     reward += -(distance_to_goal ** 3)/10
    #
    #
    #
    #     # print(progress_reward)
    #     if distance_to_goal <= 18 and self.reached_18 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_18 = True
    #         print("Reached 18m")
    #         reward += 5000
    #     elif self.reached_18 is True:
    #         remaining_distance_penalty = -15.0 * (distance_to_goal - 18)
    #         reward += remaining_distance_penalty
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -50.0 * (distance_to_goal - 18)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 15 and self.reached_15 is False:
    #         # Reward for reaching goal
    #         self.reached_15 = True
    #         print("Reached 15m")
    #         reward += 5000
    #     elif self.reached_15:
    #         remaining_distance_penalty = -20.0 * (distance_to_goal - 15)
    #         reward += remaining_distance_penalty
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -30.0 * (distance_to_goal - 15)
    #         reward += remaining_distance_penalty
    #
    #     if distance_to_goal <= 12 and self.reached_12 is False:
    #
    #         # Reward for reaching goal
    #         self.reached_12 = True
    #         print("Reached 12m")
    #         reward += 7500
    #     elif self.reached_12:
    #         remaining_distance_penalty = -25.0 * (distance_to_goal - 12)
    #         reward += remaining_distance_penalty
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -20.0 * (distance_to_goal - 12)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 9 and self.reached_9 is False:
    #         # Reward for reaching goal
    #         self.reached_9 = True
    #         print("Reached 9m")
    #         reward += 7500
    #     elif self.reached_9:
    #         remaining_distance_penalty = -20.0 * (distance_to_goal - 9)
    #         reward += remaining_distance_penalty
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -7.0 * (distance_to_goal - 9)
    #         reward += remaining_distance_penalty
    #     if distance_to_goal <= 6 and self.reached_6 is False:
    #         # Reward for reaching goal
    #         self.reached_6 = True
    #         print("Reached 6m")
    #         reward += 7500
    #     elif self.reached_6:
    #         remaining_distance_penalty = -20.0 * (distance_to_goal - 6)
    #         reward += remaining_distance_penalty
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -5.0 * (distance_to_goal - 6)
    #         reward += remaining_distance_penalty
    #
    #     if distance_to_goal <= 3 and self.reached_3 is False:
    #         # Reward for reaching goal
    #         self.reached_3 = True
    #         print("Reached 3m")
    #         reward += 7500
    #     elif self.reached_3:
    #         remaining_distance_penalty = -20.0 * (distance_to_goal - 3)
    #         reward += remaining_distance_penalty
    #
    #     else:
    #         # Penalize remaining distance to goal
    #         remaining_distance_penalty = -5.0 * (distance_to_goal - 3)
    #         reward += remaining_distance_penalty
    #
    #     if distance_to_goal > self.car_env.road_length:
    #         # print(distance_to_goal)
    #         reward += -(distance_to_goal ** 2) / 20
    #
    #     #action_magnitude = np.linalg.norm(action)
    #     # exploration_bonus = 5.0 * (1.0 - abs(action[0]))  # Encourage varying steering
    #     # exploration_bonus += 5.0 * action[1]  # Encourage forward movement
    #     # reward += exploration_bonus
    #     # print(scaled_improvement, distance_to_goal)
    #     #
    #     # print(scaled_improvement)
    #
    #     # distance_improvement is always -ve
    #
    #     pos_reward = 0
    #     if distance_improvement > self.progress_threshold:
    #
    #         # print("heyyyy")
    #         self.steps_without_progress = 0
    #         # pos_reward = 1000
    #         self.previous_distance_to_goal = distance_to_goal
    #     else:
    #
    #         self.steps_without_progress += 1
    #
    #     # End episode if stuck for too long
    #     # if self.steps_without_progress >= self.max_steps_without_progress:
    #     #     #print(dict_env['car_position'], distance_to_goal)
    #     #
    #     #     reward = -(distance_to_goal**3)*100
    #     #
    #     #     print(distance_to_goal, self.total_reward + reward, self.total_time_steps)
    #     #     done = True
    #     #     return reward, done  # Additional penalty for getting stuck
    #
    #     if self.steps_without_progress >= self.max_steps_without_progress:
    #         # New penalty formula that combines:
    #         # 1. Base penalty for getting stuck
    #         # 2. Distance-based multiplier that grows exponentially
    #         # 3. Additional scaling factor based on how far along the path the car should be
    #         # base_penalty = -(distance_to_goal ** 2)  # Changed from cubic to quadratic
    #         # distance_multiplier = np.exp(
    #         #     distance_to_goal / self.car_env.road_length)  # Reduced multiplier from *2 to *1
    #         # progress_factor = (distance_to_goal / self.car_env.road_length)   # Reduced from *100 to *10
    #
    #         #reward = base_penalty * distance_multiplier * progress_factor   # Additional scaling factor
    #         reward+= -100
    #
    #         print(f"Stuck at distance: {distance_to_goal:.2f}, Total Reward: {self.total_reward:.2f}, Steps: {self.total_time_steps}, action: {action}, reward: {reward}")
    #         done = True
    #         return reward , done
    #
    #     # if distance_to_goal < self.previous_distance_to_goal:
    #
    #     # print(distance_to_goal)
    #     if distance_to_goal < 3.:
    #         print('Reached goal', self.total_reward, self.total_time_steps)
    #         reward += 50000  # Reduced from 500000 to 1000
    #         done = True
    #         return reward, done
    #
    #     if dict_env['collision']['detected']:
    #
    #         # self.car_env.print_environment_objects()
    #         reward += -100  # Reduced from -3000 to -30
    #         print('collision detected', dict_env['collision']['detected'], dict_env['collision']['direction'],
    #               dict_env['collision']['min_distance'], dict_env['collision']['id'], "action:", action, self.total_reward, reward, distance_to_goal)
    #     #     return reward, done  # Additional penalty for getting stuck
    #         done = True
    #     # total_reward = progress_reward + collision_reward + penalty_back + pos_reward
    #     self.total_reward += reward
    #     # print(total_reward)
    #
    #     return reward, done
    def _compute_reward(self, dict_env, action):
        done = False
        reward = 0

        # Get current position and calculate distance
        distance_to_goal = np.linalg.norm(np.array(dict_env['car_position']) - np.array(self.car_env.end_point))
        distance_improvement = self.previous_distance_to_goal - distance_to_goal

        # 1. Continuous Progress Reward
        progress_reward = distance_improvement * 100  # Scale up the improvement
        reward += progress_reward

        # 2. Velocity-based reward
        # forward_velocity = (action[0] + action[2]) / 2  # Average of left wheels
        # velocity_reward = 10.0 * forward_velocity if distance_improvement > 0 else 0
        # reward += velocity_reward

        # 3. Heading reward - encourage facing toward goal
        # car_pos = dict_env['car_position']
        # _, car_orn = p.getBasePositionAndOrientation(self.car_env.car_id)
        # goal_direction = np.array([self.car_env.end_point[0] - car_pos[0],
        #                            self.car_env.end_point[1] - car_pos[1]])
        # goal_direction = goal_direction / np.linalg.norm(goal_direction)
        #
        # # Convert quaternion to euler angles
        # euler = p.getEulerFromQuaternion(car_orn)
        # car_heading = np.array([np.cos(euler[2]), np.sin(euler[2])])
        # heading_alignment = np.dot(car_heading, goal_direction)
        # heading_reward = 50.0 * heading_alignment
        # reward += heading_reward

        # if distance_to_goal > self.car_env.road_length:
        #     done = True
        #     reward += -1000
        #     print(
        #         f"Stuck at distance: {distance_to_goal:.2f}, Total Reward: {self.total_reward:.2f}, Steps: {self.total_time_steps}, action: {action}")
        #     return reward, done
        milestones = [18, 17, 16, 15, 14, 12, 9, 6, 3]
        def get_last_two_passed_milestones(distance_to_goal, milestones):
            # Sort milestones in descending order
            passed_milestones = [m for m in milestones if distance_to_goal <= m]
            return passed_milestones[:3]  # Return only the last 2 passed milestones

        for milestone in milestones:
            if distance_to_goal <= milestone and not getattr(self, f'reached_{milestone}', False):
                setattr(self, f'reached_{milestone}', True)
                reward += 1000  # Reduced from 5000/7500
                print(f"Reached {milestone}m")
            elif getattr(self, f'reached_{milestone}', True):
                # Get the last 2 passed milestones
                relevant_milestones = get_last_two_passed_milestones(distance_to_goal, milestones)

                # Only apply reward if this milestone is one of the last 2 passed
                if milestone in relevant_milestones:
                    reward += 0.8 * (milestone - distance_to_goal)
            elif distance_improvement <= self.progress_threshold and getattr(self, f'reached_{milestone}', False):
                # Same check for last 2 passed milestones
                reward += -4*(distance_to_goal - milestone)  # Reduced from 5000/7500

        # 4. Milestone Rewards (reduced magnitude but kept for guidance)

        # for milestone in milestones:
        #     if distance_to_goal <= milestone and not getattr(self, f'reached_{milestone}', False):
        #         setattr(self, f'reached_{milestone}', True)
        #         reward += 1000  # Reduced from 5000/7500
        #         print(f"Reached {milestone}m")
        #     elif getattr(self, f'reached_{milestone}', True) and:
        #         reward += 1.25*( milestone - distance_to_goal )  # Reduced from 5000/7500
        #     elif distance_improvement <= self.progress_threshold and getattr(self, f'reached_{milestone}', False):
        #         reward += -3*(distance_to_goal - milestone)  # Reduced from 5000/7500

        # 5. Stuck prevention
        if distance_improvement > self.progress_threshold:
            self.steps_without_progress = 0
            self.previous_distance_to_goal = distance_to_goal
        else:
            self.steps_without_progress += 1
            #reward -= 10

        if self.steps_without_progress >= self.max_steps_without_progress:
            reward += -300
            print(f"Stuck at distance: {distance_to_goal:.2f}, Total Reward: {self.total_reward:.2f}, Steps: {self.total_time_steps}, action: {action}, reward: {reward}")
            done = True
            return reward, done

            # Handle collision
        if dict_env['collision']['detected']:
            reward += -20  # Increased penalty
            print('collision detected', dict_env['collision']['detected'], dict_env['collision']['direction'],
                                 dict_env['collision']['min_distance'], dict_env['collision']['id'], "action:", action, self.total_reward, reward, distance_to_goal)
            #done = True

            # Handle reaching goal
        if distance_to_goal < 3.0:
            reward += 10000  # Reduced from 50000
            print('Reached goal', self.total_reward)
            done = True
        #print(reward)
        self.total_reward += reward
        return reward, done

    # def _compute_reward(self, dict_env, action):
    #     done = False
    #     #print(dict_env)
    #     penalty_back = 0
    #     collision_reward = 0
    #
    #
    #     distance_to_goal = np.linalg.norm(np.array(dict_env['car_position']) - np.array(self.car_env.end_point))
    #
    #
    #
    #     # Check if making progress towards goal
    #     distance_improvement = self.previous_distance_to_goal - distance_to_goal
    #     progress_reward = -(distance_to_goal ** 3)/100 #+ (self.car_env.road_length - distance_to_goal)/3
    #     #print(progress_reward)
    #
    #     if distance_to_goal > self.car_env.road_length:
    #         #print(distance_to_goal)
    #         penalty_back = -(distance_to_goal ** 3)/10
    #
    #     # print(scaled_improvement, distance_to_goal)
    #     #
    #     # print(scaled_improvement)
    #
    #     #distance_improvement is always -ve
    #
    #     pos_reward = 0
    #     if distance_improvement > self.progress_threshold:
    #
    #         #print("heyyyy")
    #         self.steps_without_progress = 0
    #         #pos_reward = 1000
    #         self.previous_distance_to_goal = distance_to_goal
    #     else:
    #
    #         self.steps_without_progress += 1
    #
    #     # End episode if stuck for too long
    #     # if self.steps_without_progress >= self.max_steps_without_progress:
    #     #     #print(dict_env['car_position'], distance_to_goal)
    #     #
    #     #     reward = -(distance_to_goal**3)*100
    #     #
    #     #     print(distance_to_goal, self.total_reward + reward, self.total_time_steps)
    #     #     done = True
    #     #     return reward, done  # Additional penalty for getting stuck
    #
    #     if self.steps_without_progress >= self.max_steps_without_progress:
    #         # New penalty formula that combines:
    #         # 1. Base penalty for getting stuck
    #         # 2. Distance-based multiplier that grows exponentially
    #         # 3. Additional scaling factor based on how far along the path the car should be
    #         base_penalty = -(distance_to_goal ** 3)
    #         distance_multiplier = np.exp(distance_to_goal / self.car_env.road_length * 2)
    #         progress_factor = (distance_to_goal / self.car_env.road_length) * 100
    #
    #         reward = base_penalty * distance_multiplier * progress_factor
    #
    #         print(f"Stuck at distance: {distance_to_goal:.2f}, Reward: {reward:.2f}, Steps: {self.total_time_steps}")
    #         done = True
    #         return reward, done
    #
    #     #if distance_to_goal < self.previous_distance_to_goal:
    #
    #     #print(distance_to_goal)
    #     if distance_to_goal < 3.:
    #         print('Reached goal', self.total_reward, self.total_time_steps)
    #         reward = 500000  # Big reward for reaching goal
    #         done = True
    #         return reward, done
    #
    #     if dict_env['collision']['detected']:
    #         print('collision detected', dict_env['collision']['detected'], dict_env['collision']['direction'], dict_env['collision']['min_distance'], dict_env['collision']['id'])
    #         #self.car_env.print_environment_objects()
    #         collision_reward = -3000  # Penalize collision
    #
    #
    #
    #     total_reward = progress_reward + collision_reward + penalty_back + pos_reward
    #     self.total_reward += total_reward
    #     #print(total_reward)
    #
    #     return total_reward, done

    def step(self, action):
        self.total_time_steps +=1
        wheel_velocities = self._convert_actions_to_control(action)

        # Execute action in environment
        dict_env = self.car_env.step(wheel_velocities)

        # Get new observation (only LiDAR)
        next_state = self._process_lidar_data(dict_env['lidar_data'])


        # Calculate reward
        reward, done = self._compute_reward(dict_env, action)

        # Return truncated=False as the fifth value
        return next_state, reward, done, False, {}

    def render(self):
        # Gymnasium requires render() to have render_mode parameter
        pass  # PyBullet already renders the environment

    def close(self):
        p.disconnect()

    def reset(self, seed=None, options=None):
        # Gymnasium style reset with super().reset()
        #super().reset(seed=seed)
        self.total_reward = 0
        self.total_time_steps = 0
        # Reset the car environment
        lidar_data = self.car_env.reset()
        self.prev_steering = 0.0
        self.reached_17 = False
        self.reached_15 = False
        self.reached_16 = False
        self.reached_14 = False
        self.reached_18 = False
        self.reached_12 = False
        self.reached_6 = False
        self.reached_9 = False
        self.reached_3 = False
        # Reset progress tracking variable
        self.steps_without_progress = 0
        self.previous_distance_to_goal = self.car_env.road_length  # Reset to initial distance
        #print(lidar_data)

        # Get initial observation (only LiDAR)
        observation = self._process_lidar_data(lidar_data)
        #print(observation)

        return observation, {}


def test_environment():
    """Test the environment with random actions"""
    gym_car = AutonomousCarGymnasiumEnv()
    obs, _ = gym_car.reset()

    print("Initial observation shape:", obs.shape)
    print("Initial LiDAR readings:", obs)

    for i in range(10000):
        action = gym_car.action_space.sample()  # Random action
        print(f"\nStep {i + 1}")
        print("Action (steering, speed):", action)

        obs, reward, done, truncated, info = gym_car.step(action)
        print("Reward:", reward)
        print("LiDAR readings:", obs)

        if done:
            print("Episode finished after {} steps".format(i + 1))
            break

    gym_car.close()


if __name__ == "__main__":
    test_environment()