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