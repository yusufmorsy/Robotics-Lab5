"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import collections

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range1 = robot.getDevice('range-finder')
range1.enable(timestep)
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
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
#mode = 'manual' # Part 1.1: manual mode
#mode = 'planner'
mode = 'autonomous'
#mode = 'picknplace'



###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (-8.46, -4.88) # (Pose_X, Pose_Y) in meters
    end_w   = (-1, -10) # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame'
    start_x = math.floor(359*(1+start_w[0]/12))
    start_y = math.floor(359*( -start_w[1]/12))
    end_x   = math.floor(359*(1+end_w[0]/12))
    end_y   = math.floor(359*( -end_w[1]/12))
    start = (start_x, start_y) # (x, y) in 360x360 map
    end   = (end_x  , end_y  ) # (x, y) in 360x360 map

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''

        queue = collections.deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x == end[0] and y == end[1]):
                return path
            for xx, yy in ((x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1,y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1)):
                if (0 <= xx < 360 and 0 <= yy < 360 and map[xx][yy] != 1 and (xx, yy) not in visited):
                    queue.append(path + [(xx, yy)])
                    visited.add((xx, yy))

        # return nothing if the while loop fails to find
        waypoints = []
        return waypoints

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load("map.npy")

    display.setColor(int(0x000000))
    for i in range(360):
        for j in range(360):
            display.drawPixel(i, j)

    display.setColor(int(0xFFFFFF))
    for i in range(360):
        for j in range(360):
            if map[i][j] == 1:
                display.drawPixel(i, j)

    # Part 2.2: Compute an approximation of the “configuration space”
    copy = map.copy()
    buffer = 8 #radius of the filling

    for i in range(360):
        for j in range(360):
            if copy[i][j] == 1:
                for k in range(2*buffer):
                    for l in range(2*buffer):
                        if (i + k - buffer >= 0 and j + l - buffer >= 0 and i + k - buffer < 360 and j + l - buffer < 360):
                            map[i + k - buffer][j + l - buffer] = 1          
    #plt.imshow(map)
    #plt.show()

    #update the map
    # display.setColor(int(0xFFFFFF))
    # for i in range(360):
    #     for j in range(360):
    #         if map[i][j] == 1:
    #             display.drawPixel(i, j)

    # Part 2.3 continuation: Call path_planner
    waypoints = path_planner(map, start, end)

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    display.setColor(int(0xFF8800))
    for i in range(len(waypoints)):
        x = waypoints[i][0]
        y = waypoints[i][1]
        display.drawPixel(x, y)
        waypoints[i] = (-12*(1-(x/360)), -12*(y/360))

    #print(waypoints)
    np.save("path.npy", waypoints)

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy")

    display.setColor(int(0xFF8800))
    for i in range(len(waypoints)):
        x = math.floor(359*(1+waypoints[i][0]/12))
        y = math.floor(359*( -waypoints[i][1]/12))
        display.drawPixel(x, y)

state = 0 # use this to iterate through your path

if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    pass

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.

            # Set the value of each lidar reading
            xpos = round(360-abs(int(wx*30)))
            ypos = round(abs(int(wy*30)))
            if xpos >= 360:
                xpos = 359
            if ypos >= 360:
                ypos = 359
            value = map[xpos][ypos]
            map[xpos][ypos] = map[xpos][ypos] + 5e-3
            if map[xpos][ypos] > 1: 
                map[xpos][ypos] = 1

            display.setColor(int(map[xpos][ypos] * 255))
            display.drawPixel(xpos, ypos)

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
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
            # Part 1.4: Filter map and save to filesystem
            for i in range(360):
                for j in range(360):
                    map[i][j] = map[i][j] > 0.5
            np.save("map.npy", map)
            print("Map file saved")
        elif key == ord('L'):
            map = np.load("map.npy")
            print("Map loaded")

            display.setColor(int(0x000000))
            for i in range(360):
                for j in range(360):
                    display.drawPixel(i, j)

            display.setColor(int(0xFFFFFF))
            for i in range(360):
                for j in range(360):
                    if map[i][j] == 1:
                        display.drawPixel(i, j)
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else:  # non-manual modes
        if mode == 'autonomous':
            # Path Following: Use the IK-based Feedback Controller
            if state < len(waypoints):
                # Get current target waypoint (waypoints are in world coordinates, in meters)
                goal = waypoints[state]
                goal_x, goal_y = goal

                # Compute error between current pose and goal
                dx = goal_x - pose_x
                dy = goal_y - pose_y
                rho = math.sqrt(dx**2 + dy**2)  # Distance error

                # Compute the angle from the robot to the waypoint
                goal_angle = math.atan2(dy, dx)
                # Compute the heading error (normalize between -pi and pi)
                alpha = goal_angle - pose_theta
                alpha = (alpha + math.pi) % (2*math.pi) - math.pi

                # If the robot is close enough to the current waypoint, go to the next one.
                if rho < 0.2:  # threshold in meters; adjust as needed
                    state += 1
                    # Optionally, you could slow down or stop the robot briefly here.
                else:
                    # Controller gains (tuning these gains will affect responsiveness and smoothness)
                    k_rho = 1.0
                    k_alpha = 2.0

                    # Compute the desired linear and angular velocities
                    v = k_rho * rho
                    omega = k_alpha * alpha

                    # Convert from (v, omega) to individual wheel speeds.
                    # Note: When vL and vR are set to MAX_SPEED, the robot moves at MAX_SPEED_MS.
                    vL = (v - (AXLE_LENGTH/2.0) * omega) * (MAX_SPEED / MAX_SPEED_MS)
                    vR = (v + (AXLE_LENGTH/2.0) * omega) * (MAX_SPEED / MAX_SPEED_MS)

                    # Saturate wheel speeds to be within [-MAX_SPEED, MAX_SPEED]
                    vL = max(min(vL, MAX_SPEED), -MAX_SPEED)
                    vR = max(min(vR, MAX_SPEED), -MAX_SPEED)
            else:
                # If all waypoints have been reached, stop the robot.
                vL = 0
                vR = 0

        elif mode == 'picknplace':
            # Part 4: Implement pick and place logic if needed.
            pass

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass