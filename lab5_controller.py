"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d  # Uncomment if you want to use something else for finding the configuration space
import heapq

MAX_SPEED = 7.0         # [rad/s]
MAX_SPEED_MS = 0.633    # [m/s]
AXLE_LENGTH = 0.4044    # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75  # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
              "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts = []

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

_range = robot.getDevice('range-finder')
_range.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x = 0
pose_y = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = []  # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83]  # Only keep lidar readings not blocked by robot chassis

# map will be used to hold occupancy values (0 = free, 1 = obstacle)
map = np.zeros(shape=[360, 360])
waypoints = []

##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. For autonomous operation, we set the mode to 'autonomous'
# mode = 'manual'      # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'
# mode = 'picknplace'

###################
#
# Planner Mode (skipped in autonomous mode)
#
###################
if mode == 'planner':
    try:
        map = np.load("map.npy")
        print("Map loaded from disk.")
    except Exception as e:
        print("No map file found, using default map with sample obstacles.")
        map = np.zeros((360, 360))
        map[100:150, 100:150] = 1
        map[200:250, 50:100] = 1

    kernel = np.ones((5, 5))
    config_space = convolve2d(map, kernel, mode='same')
    config_space = (config_space > 0).astype(np.uint8)
    
    plt.imshow(config_space, cmap='gray')
    plt.title("Configuration Space")
    plt.savefig("config_space.png")
    plt.close()

    start_w = (1.0, 1.0)
    end_w = (10.0, 10.0)

    def world_to_map_coords(world_coord):
        x, y = world_coord
        col = int(x * 30)
        row = 360 - int(y * 30)
        return (row, col)

    start = world_to_map_coords(start_w)
    end = world_to_map_coords(end_w)

    def heuristic(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def get_neighbors(cell, shape):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for d in directions:
            nr = cell[0] + d[0]
            nc = cell[1] + d[1]
            if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                neighbors.append((nr, nc))
        return neighbors
    
    def path_planner(c_space, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        closed_set = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            for neighbor in get_neighbors(current, c_space.shape):
                if c_space[neighbor[0], neighbor[1]] != 0:
                    continue
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    path = path_planner(config_space, start, end)
    if path is None:
        print("No path found!")
        waypoints = []
    else:
        waypoints = []
        for cell in path:
            row, col = cell
            world_x = col / 30.0
            world_y = (360 - row) / 30.0
            waypoints.append((world_x, world_y))
        np.save("path.npy", np.array(waypoints))
        print("Path saved with", len(waypoints), "waypoints.")
        
        plt.imshow(config_space, cmap='gray')
        path_x = [cell[1] for cell in path]
        path_y = [cell[0] for cell in path]
        plt.plot(path_x, path_y, color='red')
        plt.title("Planned Path")
        plt.savefig("planned_path.png")
        plt.close()
    
    while robot.step(timestep) != -1:
        pass

###################
#
# Map Initialization & Autonomous Mode
#
###################
if mode == 'autonomous':
    try:
        waypoints = np.load("path.npy").tolist()
        print("Path loaded from disk with", len(waypoints), "waypoints.")
    except Exception as e:
        print("No saved path found. Using default dummy waypoints.")
        waypoints = [(2.0, 2.0), (4.0, 4.0), (6.0, 4.0), (8.0, 6.0), (10.0, 8.0)]

state = 0  # used to iterate through the path

###################
#
# Main Control Loop (Autonomous Mode)
#
###################
while robot.step(timestep) != -1:

    ###################
    #
    # Mapping: Update display with sensor readings
    #
    ###################
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2])) - 1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        rx = math.cos(alpha) * rho
        ry = -math.sin(alpha) * rho
        t = pose_theta + np.pi/2.
        wx = math.cos(t) * rx - math.sin(t) * ry + pose_x
        wy = math.sin(t) * rx + math.cos(t) * ry + pose_y

        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            display.setColor(int(0x0000FF))
            display.drawPixel(360 - abs(int(wx * 30)), abs(int(wy * 30)))

    display.setColor(int(0xFF0000))
    display.drawPixel(360 - abs(int(pose_x * 30)), abs(int(pose_y * 30)))

    ###################
    #
    # Controller: Autonomous Feedback Control
    #
    ###################
    if len(waypoints) > 0 and state < len(waypoints):
        target = waypoints[state]  # Target waypoint (world coordinates)
        error_x = target[0] - pose_x
        error_y = target[1] - pose_y
        rho = math.sqrt(error_x**2 + error_y**2)
        desired_theta = math.atan2(error_y, error_x)
        alpha = desired_theta - pose_theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        
        k_rho = 1.0    # Gain for distance error
        k_alpha = 2.0  # Gain for angular error

        v = k_rho * rho
        omega = k_alpha * alpha

        vL = v - (AXLE_LENGTH / 2.0) * omega
        vR = v + (AXLE_LENGTH / 2.0) * omega

        max_wheel_speed = max(abs(vL), abs(vR))
        if max_wheel_speed > MAX_SPEED:
            vL = (vL / max_wheel_speed) * MAX_SPEED
            vR = (vR / max_wheel_speed) * MAX_SPEED

        if rho < 0.2:
            state += 1
            print("Reached waypoint", state)
    else:
        vL = 0
        vR = 0

    ###################
    #
    # Odometry Update
    #
    ###################
    pose_x += (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.cos(pose_theta)
    pose_y -= (vL + vR) / 2 / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0 * math.sin(pose_theta)
    pose_theta += (vR - vL) / AXLE_LENGTH / MAX_SPEED * MAX_SPEED_MS * timestep / 1000.0

    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

while robot.step(timestep) != -1:
    pass
