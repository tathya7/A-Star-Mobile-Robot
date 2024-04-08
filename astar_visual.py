#!/usr/bin/env python3


import numpy as np
import cv2
import heapq
import time
from math import dist, sqrt
import matplotlib.pyplot as plt


# Define canvas dimensions
CANVAS_WIDTH = 6000
CANVAS_HEIGHT = 2000

# Defining thresholds
dist_threshold = 10
ang_threshold = 15


################################################# INPUT FOR CLEARANCE #################################################
CLEARANCE = int(input("Enter the Clearance for the obstacle space:"))


# Robot wheel radius in mm
ROBOT_WHEEL_RADIUS = 33

# Robot body radius.
ROBOT_RADIUS = 220

# Distance between the wheels.
BASE_LENGTH = 287

CLEARANCE = CLEARANCE + ROBOT_RADIUS

RPM1 = int(input("Enter the RPM for Left Wheel:"))
RPM2 = int(input("Enter the RPM for Right Wheel:"))


actionset = [(0, RPM1), (RPM1, 0), (RPM1, RPM1), (0, RPM2),(RPM2, 0), (RPM2, RPM2), (RPM1, RPM2), (RPM2, RPM1)]

###################################### DEFINING CANVAS #################################################


# Creating clearance boundary
canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)

# Create a blank canvas
cv2.rectangle(canvas, (CLEARANCE, CLEARANCE), (CANVAS_WIDTH - CLEARANCE, CANVAS_HEIGHT - CLEARANCE), (0, 0, 0), -1)

# Define rectangle parameters
rect_width = 250
rect_height = 1000

# Top left rectangle coordinates
r1x1 = 1500
r1y1 = 0

# Bottom Right rectangel coordinates
r1x2 = r1x1 + rect_width
r1y2 = r1y1 + rect_height

# Top left clearance coordinates
clear_r1x1 = r1x1 - CLEARANCE
clear_r1y1 = r1y1

# Bottom right clearance coordinates
clear_r1x2 = r1x2 + CLEARANCE
clear_r1y2 = r1y2 + CLEARANCE

# Top left rectangle 2 coordinates
r2x1 = 2500
r2y1 = CANVAS_HEIGHT - rect_height

# Bottom right rectangle 2 coordinates
r2x2 = r2x1 + rect_width
r2y2 = CANVAS_HEIGHT

# Top left clearance coordinates
clear_r2x1 = r2x1 - CLEARANCE
clear_r2y1 = r2y1 - CLEARANCE

# Bottom right clearance coordinates
clear_r2x2 = r2x2 + CLEARANCE
clear_r2y2 = r2y2

# Define circle parameters
x_center = 4200
y_center = 800
radius = 600

# Clearance circle radius
clear_radius = radius + CLEARANCE

# Drawing clearance RECT 1
canvas[clear_r1y1:clear_r1y2, clear_r1x1:clear_r1x2] = (255, 255, 255)

# Drawing RECT 1
canvas[r1y1:r1y2, r1x1:r1x2] = (0, 180, 0)

# Clearance RECT 2
canvas[clear_r2y1:clear_r2y2, clear_r2x1:clear_r2x2] = (255, 255, 255)

# RECT 2
canvas[r2y1:r2y2, r2x1:r2x2] = (0, 180, 0)

# Clear CIRCLE
for x in range(x_center - clear_radius, x_center + clear_radius):
    for y in range(y_center - clear_radius, y_center + clear_radius):
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= clear_radius ** 2:
            canvas[y, x] = (255, 255, 255)

# CIRCLE
for x in range(x_center - radius, x_center + radius):
    for y in range(y_center - radius, y_center + radius):
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
            canvas[y, x] = (0, 180, 0)

# Resizing the canvas to display
width_small = int(CANVAS_WIDTH / 4)
height_small = int(CANVAS_HEIGHT / 4)
canvas_small = cv2.resize(canvas, (width_small, height_small))

################################# DEFINING GOAL CHECK  #############################################

# Checks if the node is goal node or not
def goal_check(x_curr, y_curr, th_curr, x_tar, y_tar, th_tar):
    # if np.sqrt((x_curr-x_tar)**2 + (y_curr-y_tar)**2) <=20 and abs(th_curr-th_tar) <= theta_threshold:
    if np.sqrt((x_curr-x_tar)**2 + (y_curr-y_tar)**2) < 10:
        return True
    else:
        return False

################################# DEFINING MODIFY VALUE FUNCTION #############################################

# To convert it into visited node space representation
def modify_value(elem, thresh):
    modified = int(round(elem*2)/2)/thresh
    return int(modified)


################################# DEFINING COST FUNCTION #############################################
def cost2move(x, y, theta, rpm_left, rpm_right):
    # Initialize time and distance
    t = 0
    distance = 0

    # Store initial position and orientation
    x_old = x
    y_old = y
    th_old = theta

    # Initialize variables for current position and orientation
    x1 = x
    y1 = y
    th1 = np.deg2rad(theta)  # Convert initial angle from degrees to radians
    dt = 0.1  # Time step for simulation

    # Calculate angular velocities of left and right wheels in radians per second
    ang_vel_l = 2 * np.pi * rpm_left / 60
    ang_vel_r = 2 * np.pi * rpm_right / 60

    # Main simulation loop running for 0.6 seconds
    while t < 0.6:
        t = t + dt  # Increment time by time step

        # Calculate incremental changes in position and orientation using kinematic model
        delta_x = ROBOT_WHEEL_RADIUS / 2 * (ang_vel_l + ang_vel_r) * np.cos(th1)
        delta_y = ROBOT_WHEEL_RADIUS / 2 * (ang_vel_l + ang_vel_r) * np.sin(th1)
        delta_th = (ROBOT_WHEEL_RADIUS / BASE_LENGTH) * (ang_vel_r - ang_vel_l)

        # Update current position and orientation
        x1 = x1 + (delta_x * dt)
        y1 = y1 + (delta_y * dt)
        th1 = th1 + np.rad2deg(delta_th) * dt  # Convert incremental angle to degrees

        # Checking if the new position is occupied by an obstacle.
        if canvas[int(round(y1 * 2) / 2), int(round(x1 * 2) / 2), 1] == 0:
            # If not updating the total distance traveled
            distance += sqrt((delta_x * dt) ** 2 + (delta_y * dt) ** 2)
            t = t + dt

        else:
            # If the new position is occupied, revert back to the old position and orientation
            x1, y1, th1 = x_old, y_old, th_old
            break  # Exit the loop

    # Return the final position, orientation, distance traveled, and wheel velocities
    return np.round(x1, 3), np.round(y1, 3), np.round(th1, 2), distance, rpm_left, rpm_right



################################################# USER INPUT COORDINATES #################################################

check = True

while check:
    x_start = int(input("Enter the initial X position ({} to {}): ".format(0+CLEARANCE, CANVAS_WIDTH-CLEARANCE-1)))
    y_start = int(input("Enter the initial Y Position ({} to {}): ".format(0+CLEARANCE, CANVAS_HEIGHT-CLEARANCE-1)))
    th_start = int(input("Enter the initial Orientation (0 to 360): "))
    print("Your Start Node Is (X,Y,Angle): ", x_start, y_start, th_start)
    # Converting the coordinates to the instructed coordinate system
    y_start = CANVAS_HEIGHT - y_start - 1 
    # Checks if the given node is in the free space
    if canvas[y_start, x_start,1] == 0:
        cv2.circle(canvas,(x_start, y_start), 2, (0, 180, 0), -1)
        check = False
    # If the starting node is in obstacle space
    else:
        print("Starting Position is in the Obstacle space! Re-Enter the Position")

check = True

while check:
    x_goal = int(input("Enter the Destination X Position ({} to {}): ".format(0+CLEARANCE, CANVAS_WIDTH-CLEARANCE-1)))
    y_goal = int(input("Enter the Destination Y Position ({} to {}): ".format(0+CLEARANCE, CANVAS_HEIGHT-CLEARANCE-1)))
    th_goal= int(input("Enter the Goal Orientation (0 to 360): "))

    print("Your Goal Node Is (X,Y,Angle): ", x_goal, y_goal, th_goal)
    # Converting the coordinates to the instructed coordinate system
    y_goal = CANVAS_HEIGHT - y_goal - 1 
    # Checks if the given node is in the free space
    if canvas[y_goal, x_goal,1] == 0:
        # Checks if the start node and goal node are same
        if (x_start, y_start) == (x_goal, y_goal):
            print("Error! Start and Goal Position Cannot be Same")
        else:
            check = False 
    else:
        print("Goal Position is in the Obstacle space! Re-Enter the Position")


start = time.time()

q = []
heapq.heappush(q,(0,x_start,y_start,th_start, actionset[0][0],actionset[0][1]))
child2parent = {}

# Modify the initial values to the visited space
x_start_mod = modify_value(x_start,dist_threshold)
y_start_mod = modify_value(y_start,dist_threshold)
th_start_mod = modify_value(th_start, ang_threshold)

#Initializing visited
visited = {(x_start_mod, y_start_mod, th_start_mod):True}
node_cost = {(x_start_mod, y_start_mod, th_start_mod):0} 
cost2come = {(x_start_mod, y_start_mod, th_start_mod):0}

print("Generating Path......")
reached = False

while q:
    cst, x_pos, y_pos,th, rpm_pos_l, rpm_pos_r = heapq.heappop(q)

    x_mod = modify_value(x_pos,dist_threshold)
    y_mod = modify_value(y_pos,dist_threshold)
    th_mod = modify_value(th,ang_threshold)

    prev_cst = cost2come[(x_mod, y_mod, th_mod)] 

    if goal_check(x_pos, y_pos,th, x_goal, y_goal, th_goal) == True:
        print("Goal Reached", x_pos, y_pos, th, rpm_pos_l, rpm_pos_r)
        reached = True
        break

    for action in actionset:
        node = cost2move(x_pos, y_pos, th, action[0], action[1])
        if node is not None:

            new_x, new_y, new_th, action_cst, rpm_left, rpm_right = node

            # Converting it to int to plot in map
            x_map = int(round(new_x*2)/2)
            y_map = int(round(new_y*2)/2)
            th_map = int(round(new_th*2)/2)


            if 0 <= new_x < CANVAS_WIDTH-1 and 0 <= new_y < CANVAS_HEIGHT-1 and canvas[y_map, x_map, 1] == 0:

                # Modifying for visited node space
                newx_mod = modify_value(x_map, dist_threshold)
                newy_mod = modify_value(y_map, dist_threshold)
                newth_mod = modify_value(th_map, ang_threshold)

                if (newx_mod, newy_mod, newth_mod) not in visited:

                    cost2come[(newx_mod,newy_mod,newth_mod)] = prev_cst + action_cst

                    node_cost[(newx_mod,newy_mod,newth_mod)] = cost2come[(newx_mod,newy_mod,newth_mod)] + dist((new_x, new_y), (x_goal, y_goal))

                    heapq.heappush(q,(node_cost[(newx_mod,newy_mod,newth_mod)], new_x, new_y, new_th, rpm_left,rpm_right))

                    child2parent[(new_x, new_y,new_th,rpm_left, rpm_right)] = (x_pos, y_pos, th, rpm_pos_l, rpm_pos_r)
                    visited[(newx_mod,newy_mod,newth_mod)] = True

                if cost2come[(newx_mod,newy_mod,newth_mod)] > prev_cst + action_cst :
                    cost2come[(newx_mod,newy_mod,newth_mod)] = prev_cst + action_cst
                    node_cost[(newx_mod,newy_mod,newth_mod)] = cost2come[(newx_mod,newy_mod,newth_mod)] +  dist((new_x, new_y), (x_goal, y_goal))
                    child2parent[(new_x, new_y,new_th,rpm_left, rpm_right)] = (x_pos, y_pos, th, rpm_pos_l, rpm_pos_r)

end = time.time()


# If the goal is not reachable
if reached == False:
    print("Goal out of bounds")
    
#  Printing the runtime of the algorithm
print("Generating Video..., Algorithm Time is: ", (end-start))



######################################### GENERATING OPTIMAL PATH #################################################

path = []

path.append((x_pos, y_pos, th,rpm_pos_l, rpm_pos_r))
x,y, th, rpml, rpmr = x_pos, y_pos, th, rpm_pos_l, rpm_pos_r
while (x,y,th,rpml, rpmr) in child2parent:

    path.append((x,y, th,rpml, rpmr))
    (x,y, th, rpml, rpmr) = child2parent[(x,y, th, rpml, rpmr)]

path.append((x_start,y_start, th_start, actionset[0][0], actionset[0][1]))
path.reverse()

#################################################   GENERATING VIDEO       #################################################

# canvas_small = cv2.resize(canvas, (width_small, height_small))
path_vid = cv2.VideoWriter('a_star_path_final4.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (width_small, height_small))

cv2.circle(canvas, (int(x_start), int(y_start)), 10, (92, 11, 227), -1)
cv2.circle(canvas, (int(x_goal), int(y_goal)), 10, (0, 165, 255), -1)

num_frames = 250
j = 0

for (x_, y_, angle, rl, rr) in child2parent:
    j += 1
    t = 0
    distance = 0
    x_par = x_
    y_par = y_

    x1 = x_
    y1 = y_
    th1 = np.deg2rad(angle)
    dt = 0.1

    ang_left = 2 * np.pi * rl / 60
    ang_right = 2 * np.pi * rr / 60

    while t<0.6:
        # t = t+dt
        delta_x = ROBOT_WHEEL_RADIUS/2 *(ang_left + ang_right) * np.cos(th1)
        delta_y = ROBOT_WHEEL_RADIUS/2 * (ang_left + ang_right) * np.sin(th1)
        delta_th = (ROBOT_WHEEL_RADIUS / BASE_LENGTH) * (ang_right - ang_left)

        x1 = x1 + (delta_x*dt)
        y1 = y1 + (delta_y*dt)
        th1 = th1 + np.rad2deg(delta_th) * dt

        x_pix = int(round(x1*2)/2)
        y_pix = int(round(y1*2)/2)

        if canvas[y_pix, x_pix, 1] == 0:
            cv2.line(canvas, (int(x_par), int(y_par)), (x_pix, y_pix),(225, 105, 65), 4)
            x_par, y_par = x1,y1
            t = t+dt

        else:
            break
            


    if (j == num_frames):
        canvas_s = cv2.resize(canvas, (width_small, height_small))
        cv2.imshow("Canvas", canvas_s)
        path_vid.write(canvas_s)
        cv2.waitKey(1)
        j=0

# Draw the start and goal nodes on the canvas
cv2.circle(canvas, (x_start, y_start), 10, (0, 255, 0), 20)
cv2.circle(canvas, (x_goal, y_goal), 10, (0, 165, 255), 20)

for i in range(len(path)-1):
    
    cv2.line(canvas,(int(path[i][0]),int(path[i][1])), (int(path[i+1][0]),int(path[i+1][1])), (0, 0, 255),8)
    canvas_s = cv2.resize(canvas, (width_small, height_small))
    cv2.imshow('Video', canvas_s)
    path_vid.write(canvas_s)
    cv2.waitKey(1)

last_frame = canvas_s
for _ in range(200):
    path_vid.write(last_frame)

path_vid.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
