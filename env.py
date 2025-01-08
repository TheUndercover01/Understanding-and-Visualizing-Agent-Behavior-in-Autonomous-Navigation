import pybullet as p
import pybullet_data
import numpy as np
import time
import pickle



class AutonomousCarEnv:
    def __init__(self, client = None, seed = 1):

        if client is None:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = client
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.road_length = 20
        self.plane_id = p.loadURDF("plane.urdf")
        self.seed = 1
        if seed is not None:
            np.random.seed(seed)

        self.car_id = p.loadURDF("husky/husky.urdf", [0, 0., 0.])
        print("wakanda",self.car_id)
        print(f"Connected to PyBullet with client ID: {self.client}")
        # ... rest of initialization
        print(f"Created plane with ID: {self.plane_id}")
        print(f"Created car with ID: {self.car_id}")
        # Add damping to make motion more stable
        # for joint in range(p.getNumJoints(self.car_id)):
        # p.changeDynamics(self.car_id, joint,
        #                  linearDamping=0.5,
        #                  angularDamping=0.5,
        #                  maxJointVelocity=20)

        self.lidar_height = 0.37
        #self.num_rays = 10
        self.ray_length = 4.0
        self.ray_start_height = 0.37
        self.lidar_debug_lines = []  # Store debug line IDs for updating visualization

        self.create_road()
        self.create_random_obstacles()
        self.create_obstacles()

        self.end_point = [self.road_length + 2, 0, 0]
        self.create_end_marker()
        self.terminal_text_id = None

    def create_road(self):
        road_width = 4
        barrier_height = 1.0

        barrier_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_length / 2, 0.2, barrier_height / 2],
            rgbaColor=[1, 0, 0, 1.0]
        )
        barrier_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.road_length / 2, 0.2, barrier_height / 2]
        )

        for y_pos in [road_width / 2, -road_width / 2]:
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=barrier_collision,
                baseVisualShapeIndex=barrier_shape,
                basePosition=[self.road_length / 2, y_pos, barrier_height / 2]
            )

    def create_random_obstacles(self):
        self.obstacles = []
        road_width = 4
        min_path_width = 1.5

        # Define fixed obstacle configurations for specific seeds
        if self.seed == 1:
            # Simple course with basic obstacles
            self.obstacles = [
                {"type": "cylinder", "pos": [3, -.65, 0.35], "size": [0.3, 1.5], "color": [0, 1, 0, 1]},
                {"type": "box", "pos": [7, 0., 0.65], "size": [0.25, 0.45, 1.5], "color": [1, 0, 0, 1]},
                {"type": "sphere", "pos": [13.5, 1.1, 0.4], "size": [0.4], "color": [0, 0, 1, 1]},
                #{"type": "sphere", "pos": [13.5, -1.1, 0.4], "size": [0.4], "color": [0, 0, 1, 1]},
                {"type": "cylinder", "pos": [17.5, -0.5, 0.75], "size": [0.3, 1.5], "color": [0, 1, 0, 1]},
                #{"type": "sphere", "pos": [13, 0.5, 0.4], "size": [0.4], "color": [0, 0, 1, 1]}
            ]
        elif self.seed == 2:
            # More challenging course with tighter spaces
            self.obstacles = [
                {"type": "box", "pos": [4, -1, 0.5], "size": [0.4, 0.4, 1.0], "color": [1, 0, 0, 1]},
                {"type": "box", "pos": [4, 1, 0.5], "size": [0.4, 0.4, 1.0], "color": [1, 0, 0, 1]},
                {"type": "cylinder", "pos": [9, 0, 0.75], "size": [0.3, 1.5], "color": [0, 1, 0, 1]},
                {"type": "sphere", "pos": [13, -.5, 0.4], "size": [0.4], "color": [0, 0, 1, 1]},
                {"type": "sphere", "pos": [16, 0.75, 0.4], "size": [0.4], "color": [0, 0, 1, 1]},
                {"type": "box", "pos": [21, -.25, 0.75], "size": [0.5, 0.5, 1.5], "color": [1, 1, 0, 1]},
                #{"type": "sphere", "pos": [23, -1, 0.75], "size": [0.5, 0.5, 1.5], "color": [1, 1, 0, 1]},
                #{"type": "cylinder", "pos": [22, 0.25, 0.75], "size": [0.3, 1.5], "color": [0, 1, 0, 1]},
            ]
        else:
            # Use original random generation logic with fixed seed if provided
            obstacle_types = {
                "box": {"ranges": {"x": (0.3, 0.6), "y": (0.3, 0.6), "z": (0.5, 1.5)}},
                "cylinder": {"ranges": {"radius": (0.2, 0.4), "height": (0.5, 1.5)}},
                "sphere": {"ranges": {"radius": (0.2, 0.5)}}
            }

            road_sections = [(3, 7), (8, 12), (13, 17), (18, 22)]

            for section in road_sections:
                num_obstacles = np.random.randint(1, 3)
                for _ in range(num_obstacles):
                    valid_position = False
                    while not valid_position:
                        x = np.random.uniform(section[0], section[1])
                        y = np.random.uniform(-road_width / 2 + 0.5, road_width / 2 - 0.5)

                        valid_position = True
                        for obs in self.obstacles:
                            dist = np.sqrt((x - obs["pos"][0]) ** 2 + (y - obs["pos"][1]) ** 2)
                            if dist < min_path_width:
                                valid_position = False
                                break

                    obs_type = np.random.choice(list(obstacle_types.keys()))
                    color = [np.random.random(), np.random.random(), np.random.random(), 1]

                    if obs_type == "box":
                        ranges = obstacle_types["box"]["ranges"]
                        size = [
                            np.random.uniform(*ranges["x"]),
                            np.random.uniform(*ranges["y"]),
                            np.random.uniform(*ranges["z"])
                        ]
                        self.obstacles.append({
                            "type": "box",
                            "pos": [x, y, size[2] / 2],
                            "size": size,
                            "color": color
                        })
                    elif obs_type == "cylinder":
                        ranges = obstacle_types["cylinder"]["ranges"]
                        radius = np.random.uniform(*ranges["radius"])
                        height = np.random.uniform(*ranges["height"])
                        self.obstacles.append({
                            "type": "cylinder",
                            "pos": [x, y, height / 2],
                            "size": [radius, height],
                            "color": color
                        })
                    elif obs_type == "sphere":
                        radius = np.random.uniform(*obstacle_types["sphere"]["ranges"]["radius"])
                        self.obstacles.append({
                            "type": "sphere",
                            "pos": [x, y, radius],
                            "size": [radius],
                            "color": color
                        })

    def create_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle["type"] == "box":
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=obstacle["size"],
                    rgbaColor=obstacle["color"]
                )
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=obstacle["size"]
                )
            elif obstacle["type"] == "cylinder":
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=obstacle["size"][0],
                    length=obstacle["size"][1],
                    rgbaColor=obstacle["color"]
                )
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=obstacle["size"][0],
                    height=obstacle["size"][1]
                )
            elif obstacle["type"] == "sphere":
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=obstacle["size"][0],
                    rgbaColor=obstacle["color"]
                )
                collision_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=obstacle["size"][0]
                )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obstacle["pos"]
            )

    def create_end_marker(self):
        # Large green cylinder as end marker
        marker_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=2.0,
            length=2.0,
            rgbaColor=[0, 1, 0, 0.7]
        )
        marker_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=2.0,
            height=2.0
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=marker_collision,
            baseVisualShapeIndex=marker_shape,
            basePosition=self.end_point
        )

    def show_terminal_state(self):
        if self.terminal_text_id is not None:
            p.removeUserDebugItem(self.terminal_text_id)
        self.terminal_text_id = p.addUserDebugText(
            "GOAL REACHED!",
            [self.end_point[0], self.end_point[1], self.end_point[2] + 2],
            [0, 1, 0],
            textSize=2.0
        )

    def reset(self):
        if self.terminal_text_id is not None:
            p.removeUserDebugItem(self.terminal_text_id)
        p.resetBasePositionAndOrientation(self.car_id, [0, 0, 0.], [0, 0, 0, 1])
        return self.get_lidar_data()

    def detect_collision(self, lidar_data, baseline_data):
        points_per_lidar = 4 * 8
        MIN_SAFE_DISTANCE = 0.15

        lidar_sections = {
            'Front': slice(0, points_per_lidar),
            'Right': slice(points_per_lidar, 2 * points_per_lidar),
            'Back': slice(2 * points_per_lidar, 3 * points_per_lidar),
            'Left': slice(3 * points_per_lidar, 4 * points_per_lidar)
        }

        id = None
        collision_detected = False
        collision_direction = None

        min_distance = float('inf')

        for direction, slice_range in lidar_sections.items():
            section_data = lidar_data[slice_range]

            for current in section_data:
                # Skip if hit object is floor (ID 0) or car (ID 1)
                if current[0] in [0, self.car_id]:
                    continue

                current_distance = current[2] * self.ray_length

                if current_distance < MIN_SAFE_DISTANCE:
                    collision_detected = True
                    collision_direction = direction
                    id = current[0]
                    min_distance = min(min_distance, current_distance)
                    break

        return collision_detected, collision_direction, min_distance, id

    def get_lidar_data(self, visualize=False):
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id)
        ray_starts = []
        ray_ends = []

        # LiDAR mounting positions relative to car center (front, right, back, left)
        lidar_positions = [
            [0.45, 0, 0],  # Front
            [0, -0.36, -0.1],  # Right
            [-0.45, 0, 0],  # Back
            [0, 0.36, -0.1]  # Left
        ]

        # Base directions for each LiDAR (facing front, right, back, left)
        base_directions = [
            [0, 0],  # Front: 0° rotation
            [-np.pi / 2, 0],  # Right: 90° rotation
            [np.pi, 0],  # Back: 180° rotation
            [np.pi / 2, 0]  # Left: -90° rotation
        ]

        num_vertical_layers = 4
        num_horizontal_points = 8

        # Generate rays for each LiDAR
        for lidar_pos, base_dir in zip(lidar_positions, base_directions):
            # Vertical angles from -45° to 45° (half hemisphere)
            phi = np.linspace(-np.pi /4, np.pi / 4, num_vertical_layers)
            # Horizontal angles from -90° to 90° (half hemisphere)
            theta = np.linspace(-np.pi / 2, np.pi / 2, num_horizontal_points)

            # Create meshgrid for all combinations
            Phi, Theta = np.meshgrid(phi, theta)
            phi_points = Phi.flatten()
            theta_points = Theta.flatten()

            # Adjust ray direction based on LiDAR position
            base_yaw, base_pitch = base_dir

            for phi, theta in zip(phi_points, theta_points):
                # Rotate the ray direction based on LiDAR orientation
                adjusted_theta = theta + base_yaw

                # Convert to Cartesian coordinates
                x = self.ray_length * np.cos(phi) * np.cos(adjusted_theta)
                y = self.ray_length * np.cos(phi) * np.sin(adjusted_theta)
                z = self.ray_length * np.sin(phi)

                # Adjust start position based on LiDAR mounting point
                ray_start = [
                    car_pos[0] + lidar_pos[0],
                    car_pos[1] + lidar_pos[1],
                    car_pos[2] + self.ray_start_height + lidar_pos[2]
                ]

                ray_end = [
                    ray_start[0] + x,
                    ray_start[1] + y,
                    ray_start[2] + z
                ]

                ray_starts.append(ray_start)
                ray_ends.append(ray_end)

        results = p.rayTestBatch(ray_starts, ray_ends)
        if visualize:
            self.visualize_lidar(ray_starts, ray_ends, results)

        # if True:
        #     #self.visualize_lidar(ray_starts, ray_ends, results)
        #
        #     # Print detailed LiDAR readings for each sensor
        #     lidar_names = ['Front', 'Right', 'Back', 'Left']
        #     points_per_lidar = num_vertical_layers * num_horizontal_points
        #
        #     for lidar_idx, lidar_name in enumerate(lidar_names):
        #         print(f"\n{lidar_name} LiDAR Readings:")
        #         print("Vertical (°) | Horizontal (°) | Distance (m) | Hit Object ID | Hit Position")
        #         print("-" * 75)
        #
        #         start_idx = lidar_idx * points_per_lidar
        #         end_idx = start_idx + points_per_lidar
        #
        #         for i in range(start_idx, end_idx):
        #             result = results[i]
        #             phi = phi_points[i % points_per_lidar]
        #             theta = theta_points[i % points_per_lidar]
        #
        #             hit_fraction = result[2]
        #             distance = hit_fraction * self.ray_length
        #             hit_object_id = result[0]
        #             hit_position = result[3]
        #
        #             vertical_deg = np.degrees(phi)
        #             horizontal_deg = np.degrees(theta)
        #
        #             if (i - start_idx) % num_horizontal_points == 0:
        #                 print("-" * 75)
        #
        #             print(
        #                 f"{vertical_deg:11.1f} | {horizontal_deg:13.1f} | {distance:11.2f} | {hit_object_id:11d} | {hit_position}")

        return results
    def visualize_lidar(self, ray_starts, ray_ends, results):
        # Remove previous debug lines
        for line_id in self.lidar_debug_lines:
            p.removeUserDebugItem(line_id)
        self.lidar_debug_lines.clear()

        # Draw new debug lines for each ray
        for i in range(len(results)):
            hit_fraction = results[i][2]
            if hit_fraction < 1.0:  # Ray hit something
                hit_position = [
                    ray_starts[i][0] + (ray_ends[i][0] - ray_starts[i][0]) * hit_fraction,
                    ray_starts[i][1] + (ray_ends[i][1] - ray_starts[i][1]) * hit_fraction,
                    ray_starts[i][2] + (ray_ends[i][2] - ray_starts[i][2]) * hit_fraction
                ]
                # Red line for hits
                line_id = p.addUserDebugLine(ray_starts[i], hit_position, [1, 0, 0])
            else:
                # Green line for no hits
                line_id = p.addUserDebugLine(ray_starts[i], ray_ends[i], [0, 1, 0])
            self.lidar_debug_lines.append(line_id)
        # self.visualize_lidar(ray_starts, ray_ends, results)

    # def step(self, action):
    #     for i, wheel in enumerate([2, 3, 4, 5]):
    #         p.setJointMotorControl2(self.car_id, wheel,
    #                                 p.VELOCITY_CONTROL,
    #                                 targetVelocity=action[i],
    #                                 force=20.0)
    #
    #     p.stepSimulation()
    #
    #
    #     # Add visualize parameter to control ray visualization
    #     lidar_data = self.get_lidar_data(visualize=True)  # Set to False to disable visualization
    #     car_pos, _ = p.getBasePositionAndOrientation(self.car_id)
    #
    #     # print(f"\nCar Position: x={car_pos[0]:.2f}, y={car_pos[1]:.2f}, z={car_pos[2]:.2f}")
    #     # for i, ray in enumerate(lidar_data):
    #     #     if i % 10 == 0:  # Print every 10th ray
    #     #         angle = 360 * i / len(lidar_data)
    #     #         distance = ray[2] * env.ray_length
    #     #         print(f"Angle {angle:.1f}°: {distance:.2f}m")
    #
    #     distance_to_goal = np.linalg.norm(np.array(car_pos) - np.array(self.end_point))
    #     done = distance_to_goal < 1.0
    #
    #     if done:
    #         self.show_terminal_state()
    #
    #     return lidar_data, car_pos, done
    def step(self, action):
        for i, wheel in enumerate([2, 3, 4, 5]):
            p.setJointMotorControl2(
                self.car_id, wheel,
                p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=20.0
            )

        p.stepSimulation()

        lidar_data = self.get_lidar_data(visualize=False)

        # Load baseline data (should be loaded once in __init__)
        with open("lidar_baseline.pkl", "rb") as f:
            baseline_data = pickle.load(f)

        collision_detected, direction, distance, id = self.detect_collision(lidar_data, baseline_data)
        car_pos, _ = p.getBasePositionAndOrientation(self.car_id)
        # distance_to_goal = np.linalg.norm(np.array(car_pos) - np.array(self.end_point))
        # done = distance_to_goal < 1.0
        #
        # if done:
        #     if collision_detected:
        #         print(f"Collision detected on {direction} side! Distance: {distance:.2f}m")
        #     else:
        #         self.show_terminal_state()

        return {
            'lidar_data': lidar_data,
            'car_position': car_pos,
            'collision': {
                'detected': collision_detected,
                'direction': direction,
                'min_distance': distance,
                'id': id
            },
            #'done': done
        }

    def get_environment_objects(self):
        """
        Get information about all objects in the PyBullet environment.
        Returns a dictionary mapping object IDs to their properties.
        """
        num_objects = p.getNumBodies()
        objects_dict = {}

        for obj_id in range(num_objects):
            # Get basic object info
            body_info = p.getBodyInfo(obj_id)
            object_name = body_info[1].decode('utf-8')  # Convert byte string to regular string

            # Get position and orientation
            pos, orn = p.getBasePositionAndOrientation(obj_id)

            # Get collision status and dynamics info
            dynamics_info = p.getDynamicsInfo(obj_id, -1)
            mass = dynamics_info[0]

            # Store object information
            objects_dict[obj_id] = {
                'name': object_name,
                'position': [round(x, 3) for x in pos],
                'mass': mass,
                'is_static': mass == 0
            }

            # Get additional info for the car (joints)
            if 'husky' in object_name.lower():
                num_joints = p.getNumJoints(obj_id)
                joints = {}
                for joint_id in range(num_joints):
                    joint_info = p.getJointInfo(obj_id, joint_id)
                    joints[joint_id] = {
                        'name': joint_info[1].decode('utf-8'),
                        'type': joint_info[2],
                    }
                objects_dict[obj_id]['joints'] = joints

        return objects_dict

    # Function to print the objects in a readable format
    def print_environment_objects(self):

        objects = self.get_environment_objects()

        print("\nObjects in Environment:")
        print("-" * 50)

        # Group objects by type for better readability
        static_objects = []
        dynamic_objects = []

        for obj_id, info in objects.items():
            if info['is_static']:
                static_objects.append((obj_id, info))
            else:
                dynamic_objects.append((obj_id, info))

        print("\nDynamic Objects:")
        for obj_id, info in dynamic_objects:
            print(f"\nObject ID: {obj_id}")
            print(f"Name: {info['name']}")
            print(f"Position: {info['position']}")
            print(f"Mass: {info['mass']}")

            if 'joints' in info:
                print("Joints:")
                for joint_id, joint_info in info['joints'].items():
                    print(f"  Joint {joint_id}: {joint_info['name']}")

        print("\nStatic Objects (Barriers, Obstacles, End Marker):")
        for obj_id, info in static_objects:
            print(f"\nObject ID: {obj_id}")
            print(f"Name: {info['name']}")
            print(f"Position: {info['position']}")





if __name__ == "__main__":
    env = AutonomousCarEnv()
    num_joints = p.getNumJoints(env.car_id)

    # Print joint info for all joints
    print(f"Number of joints: {num_joints}")
    for joint_id in range(num_joints):
        joint_info = p.getJointInfo(env.car_id, joint_id)
        print(f"Joint ID: {joint_id}")
        print(f"  Name: {joint_info[1].decode('utf-8')}")
        print(f"  Type: {joint_info[2]}")
        print(f"  Parent Link: {joint_info[16]}")
        print(f"  Child Link: {joint_info[12].decode('utf-8')}")
        print(f"  Lower Limit: {joint_info[8]}")
        print(f"  Upper Limit: {joint_info[9]}")
        print(f"  Max Force: {joint_info[10]}")
        print(f"  Max Velocity: {joint_info[11]}")
        print()

    aabb_min, aabb_max = p.getAABB(env.car_id)

    print(aabb_min, aabb_max)

    # Calculate dimensions
    dimensions = np.array(aabb_max) - np.array(aabb_min)
    print(f"Object dimensions (L x W x H): {dimensions}")

    env.print_environment_objects()

    # Disconnect from PyBullet

    a = 0
    try:
        while True:
            # Set higher velocity for forward movement
            left_wheels_velocity = 20.0
            right_wheels_velocity = 65.0  # lower velocity for right to move right

            # Control wheels with smoother acceleration

            #lidar_data, car_pos, done = env.step(
                #[left_wheels_velocity, right_wheels_velocity, left_wheels_velocity, right_wheels_velocity])
            b = env.step(
             [left_wheels_velocity, right_wheels_velocity, left_wheels_velocity, right_wheels_velocity])
            b['done'] = False

            # Get LiDAR data
            # lidar_data = env.get_lidar_data()
            # car_pos, _ = p.getBasePositionAndOrientation(env.car_id)

            # Print current position
            # print(f"\nCar Position: x={car_pos[0]:.2f}, y={car_pos[1]:.2f}, z={car_pos[2]:.2f}")

            # # Print some LiDAR readings
            # for i, ray in enumerate(lidar_data):
            #     if i % 10 == 0:  # Print every 10th ray
            #         angle = 360 * i / len(lidar_data)
            #         distance = ray[2] * env.ray_length
            #         print(f"Angle {angle:.1f}°: {distance:.2f}m")
            #
            # time.sleep(1. / 240.)
            #
            # # Check if reached end point
            # distance_to_goal = np.linalg.norm(np.array(car_pos) - np.array(env.end_point))
            if b['done']:
                print("Goal reached!")
                break

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")