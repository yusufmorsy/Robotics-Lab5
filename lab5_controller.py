"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
import collections

# --------------------------------------------------------------------------------
# Robot & Motion Constants
# --------------------------------------------------------------------------------
MAX_SPEED = 7.0       # [rad/s]
MAX_SPEED_MS = 0.633  # [m/s]
AXLE_LENGTH = 0.4044  # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75   # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# --------------------------------------------------------------------------------
# Controller Gains for Autonomous Waypoint-Following
# --------------------------------------------------------------------------------
K_RHO = 0.75               # Gain for linear velocity
K_ALPHA = 1.0             # Gain for angular velocity
DISTANCE_THRESHOLD = 0.15  # Distance at which we consider the waypoint reached

# --------------------------------------------------------------------------------
# Create the Robot instance and get basic time step
# --------------------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --------------------------------------------------------------------------------
# Tiago Robot Motors Setup
# --------------------------------------------------------------------------------
part_names = (
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
    "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint"
)
target_pos = (
    0.0, 0.0, 0.09, 0.07, 1.02,
    -3.16, 1.27, 1.32, 0.0, 1.41,
    'inf', 'inf'
)
robot_parts = []
for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# --------------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------------
range_finder = robot.getDevice('range-finder')
range_finder.enable(timestep)

camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

display = robot.getDevice("display")

# --------------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------------
pose_x = 0.0
pose_y = 0.0
pose_theta = 0.0

vL = 0.0
vR = 0.0

lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83 : len(lidar_offsets) - 83]

# --------------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------------
# mode = 'manual'
#mode = 'planner'
mode = 'autonomous'
# mode = 'picknplace'

# --------------------------------------------------------------------------------
# BFS (Planner) Implementation
# --------------------------------------------------------------------------------
def path_planner(map_data, start, end):
    queue = collections.deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if x == end[0] and y == end[1]:
            return path
        for xx, yy in (
            (x+1,y), (x-1,y), (x,y+1), (x,y-1),
            (x+1,y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1)
        ):
            if 0 <= xx < 360 and 0 <= yy < 360:
                if map_data[xx][yy] != 1 and (xx, yy) not in visited:
                    queue.append(path + [(xx, yy)])
                    visited.add((xx, yy))
    return []

# --------------------------------------------------------------------------------
# Map array, waypoints
# --------------------------------------------------------------------------------
map_array = np.zeros((360, 360))
waypoints = []
state = 0

# --------------------------------------------------------------------------------
# Mode: planner
# --------------------------------------------------------------------------------
if mode == 'planner':
    start_w = (-8.46, -4.88)
    end_w   = (-1.0, -10.0)

    start_x = math.floor(359 * (1 + start_w[0]/12))
    start_y = math.floor(359 * ( -start_w[1]/12))
    end_x   = math.floor(359 * (1 + end_w[0]/12))
    end_y   = math.floor(359 * ( -end_w[1]/12))

    start_cell = (start_x, start_y)
    end_cell   = (end_x, end_y)

    map_array = np.load("map.npy")

    display.setColor(int(0x000000))
    for i in range(360):
        for j in range(360):
            display.drawPixel(i, j)

    display.setColor(int(0xFFFFFF))
    for i in range(360):
        for j in range(360):
            if map_array[i][j] == 1:
                display.drawPixel(i, j)

    # Inflate obstacles
    copy_map = map_array.copy()
    buffer = 8
    for i in range(360):
        for j in range(360):
            if copy_map[i][j] == 1:
                for k in range(2*buffer):
                    for l in range(2*buffer):
                        ii = i + k - buffer
                        jj = j + l - buffer
                        if 0 <= ii < 360 and 0 <= jj < 360:
                            map_array[ii][jj] = 1

    path_cells = path_planner(map_array, start_cell, end_cell)

    display.setColor(int(0xFF8800))
    for (cx, cy) in path_cells:
        display.drawPixel(cx, cy)

    waypoints = []
    for (cx, cy) in path_cells:
        wx = -12*(1 - (cx / 360))
        wy = -12*(cy / 360)
        waypoints.append((wx, wy))

    np.save("path.npy", waypoints)

# --------------------------------------------------------------------------------
# Mode: autonomous
# --------------------------------------------------------------------------------
if mode == 'autonomous':
    waypoints = np.load("path.npy")
    display.setColor(int(0xFF8800))
    for wpt in waypoints:
        wptx = math.floor(359 * (1 + wpt[0]/12))
        wpty = math.floor(359 * ( -wpt[1]/12))
        display.drawPixel(wptx, wpty)
    state = 0

# --------------------------------------------------------------------------------
# Mode: picknplace
# --------------------------------------------------------------------------------
if mode == 'picknplace':
    start_ws = [(3.7, 5.7)]
    end_ws   = [(10.0, 9.3)]
    pass

# --------------------------------------------------------------------------------
# Main Loop
# --------------------------------------------------------------------------------
while robot.step(timestep) != -1 and mode != 'planner':

    # 1) GPS + compass -> pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    n = compass.getValues()

    # --- CHANGED here: remove the leading "-" and keep the offset ---
    #    i.e. rad = (math.atan2(n[0], n[2])) - 1.5708
    #    Instead of: rad = -((math.atan2(n[0], n[2])) - 1.5708)
    rad = (math.atan2(n[0], n[2])) - 1.5708  # <--- CHANGED
    pose_theta = rad

    # 2) Lidar scanning & real-time map
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83 : len(lidar_sensor_readings) - 83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]
        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue
        rx = math.cos(alpha) * rho
        ry = -math.sin(alpha) * rho

        t = pose_theta + math.pi/2.0
        wx = math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy = math.sin(t)*rx + math.cos(t)*ry + pose_y

        wx = min(wx, 11.999)
        wy = min(wy, 11.999)

        if rho < LIDAR_SENSOR_MAX_RANGE:
            xpos = round(360 - abs(int(wx * 30)))
            ypos = round(abs(int(wy * 30)))
            xpos = min(xpos, 359)
            ypos = min(ypos, 359)

            map_array[xpos][ypos] += 5e-3
            if map_array[xpos][ypos] > 1:
                map_array[xpos][ypos] = 1

            display.setColor(int(map_array[xpos][ypos] * 255))
            display.drawPixel(xpos, ypos)

    # Draw robot pose
    display.setColor(int(0xFF0000))
    display.drawPixel(360 - abs(int(pose_x * 30)), abs(int(pose_y * 30)))

    # 3) Controller
    if mode == 'manual':
        key = keyboard.getKey()
        while keyboard.getKey() != -1:
            pass
        if key == keyboard.LEFT:
            vL = -MAX_SPEED
            vR =  MAX_SPEED
        elif key == keyboard.RIGHT:
            vL =  MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Save thresholded map
            for i in range(360):
                for j in range(360):
                    map_array[i][j] = 1 if map_array[i][j] > 0.5 else 0
            np.save("map.npy", map_array)
            print("Map file saved (map.npy).")
        elif key == ord('L'):
            map_array = np.load("map.npy")
            print("Map loaded (map.npy).")

            display.setColor(int(0x000000))
            for i in range(360):
                for j in range(360):
                    display.drawPixel(i, j)
            display.setColor(int(0xFFFFFF))
            for i in range(360):
                for j in range(360):
                    if map_array[i][j] == 1:
                        display.drawPixel(i, j)
        else:
            vL *= 0.75
            vR *= 0.75

    else:
        if mode == 'autonomous':
            # 1) If we have no more waypoints, stop
            if state >= len(waypoints):
                vL = 0
                vR = 0
            else:
                # 2) Current target
                target_x, target_y = waypoints[state]
    
                # 3) Distance & heading error
                dx = target_x - pose_x
                dy = target_y - pose_y
                rho = math.sqrt(dx*dx + dy*dy)  # distance
    
                desired_heading = math.atan2(dy, dx)
                alpha = desired_heading - pose_theta  # ← the standard sign
    
                # Normalize alpha to [-pi, pi]
                alpha = (alpha + math.pi) % (2*math.pi) - math.pi
    
                # 4) Check if we reached this waypoint
                if rho < DISTANCE_THRESHOLD:
                    # Move to next waypoint
                    state += 1
                    vL = 0
                    vR = 0
                else:
                    # 5) Proportional control
                    v = K_RHO * rho
                    # Cap the forward speed, e.g. 80% of robot max
                    if v > MAX_SPEED_MS * 0.8:
                        v = MAX_SPEED_MS * 0.8
    
                    w = K_ALPHA * alpha
    
                    # 6) Convert (v,w) → left/right wheel speeds
                    WHEEL_RADIUS = MAX_SPEED_MS / MAX_SPEED  # ≈ 0.0904
                    vL_lin = v - (w * (AXLE_LENGTH / 2.0))
                    vR_lin = v + (w * (AXLE_LENGTH / 2.0))
                    
                    # Convert linear m/s to rad/s
                    vL = vL_lin / WHEEL_RADIUS
                    vR = vR_lin / WHEEL_RADIUS
    
                    # 7) Saturate wheel speeds
                    if vL >  MAX_SPEED: vL =  MAX_SPEED
                    if vL < -MAX_SPEED: vL = -MAX_SPEED
                    if vR >  MAX_SPEED: vR =  MAX_SPEED
                    if vR < -MAX_SPEED: vR = -MAX_SPEED
    
        else:
            # Other modes, e.g. picknplace
            vL = 0
            vR = 0

    # 4) Send wheel commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)

# Keep controller alive if 'planner' ends the loop
while robot.step(timestep) != -1:
    pass
