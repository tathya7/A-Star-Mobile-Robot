## Authors 
### Tathya Bhatt (tathyab | 120340246) & Mohammed Munnawar (mmunawwa | 120241642)

# A* Path Planning Algorithm for TurtleBot Navigation

In this project we implement the A* path planning algorithm to find the shortest path from the start point to the goal point while avoiding the obstacles present in the environment and eventually simulate this in a gazebo world with the generated path.

## Github Repository & Video Links

Added the TAs as collaborators and the link for the github repo is : 

Part1 Video Link : https://www.youtube.com/watch?v=tYxHOVnaW3Y
Part2 Video Link : https://youtu.be/mkmKPcqHzUI

## Requirements that were used in this 

- Python 3
- OpenCV (cv2)
- NumPy
- heapq
- math

## Steps to run Part 1:

1. Clone this repository or download the `astar_visual.py` file.
2. Ensure you have the required dependencies installed. You can install them using pip:
   
    ```bash
    pip install numpy opencv-python-headless
    ```
   
3. Run the script:

    ```bash
    python a_star_path_planning.py
    ```

4. Once you've installed the required libraries, you can run the code. Follow the prompts to input the parameters such as step size, clearance, robot radius, initial position and goal position.

5. The input robot parameters are predefined like robot radius, axial length and wheel radius.

6. The test RPMs which were used are `(25,50)`, `(30,60)`, `(50,100)`.

7. Similarly at all these RPMs, different locations were tested like from `[300,300]` to `[5700, 1700]`

8. Make sure to select positions within these ranges to ensure the path is generated correctly and to avoid obstacles.

9. Once the simulation begins, it will continue until the goal is reached. At that point, a simulation video will be created as output.

10. You can find all the necessary files and implementations in the provided GitHub repository.

#### Inputs

- RPM (Left Wheel) : For the left and right wheel velocity
- Clearance: Specifies the clearance around obstacles.
- Initial Position: X and Y coordinates along with orientation angle.
- Goal Position: X and Y coordinates.

#### Outputs

- Video: `a_star_path_final.mp4` showing the path planning process and the final path.


## Steps to Run Part 2

# Running a ROS Package

This repository contains a ROS package for [describe your package here].

## Prerequisites

Before running the ROS package, ensure you have the following prerequisites installed:

- ROS (Robot Operating System)
- [Other dependencies specific to your ROS package]

## Installation

1. Create a workspace:
   ```bash
    mkdir -p project3_ws/src
    cd ~/project3_ws/src
    ```

2. Clone the repository to your local computer:
    ```bash
    git clone https://github.com/yourusername/your-repository.git
    ```

2. Enable ROS commands:
   ```bash
    source /opt/ros/galactic/setup.bash
    ```

3. Build the ROS package using:
    ```bash
    cd ~/project3_ws
    colcon build --packages-select turtlebot3_project3
    ```

4. Source ROS (Package will be identified):
    ```bash
    source install/setup.bash
    ```

## Running

1. Launch the Gazebo environment:
    ```bash
    ros2 launch turtlebot3_project3 competition_world.launch
    ```
2. Launch the ROS package for running the turtlebot3:
   ```bash
    ros2 run turtlebot3_project3 ros_sim.py
    ```


## Additional Notes

- In the simulation the Start point and the Goal point that are feeded are START = (500, 1000) and GOAL = (2400, 500)
- The start point in this simulation is fixed at (500, 1000) but the Goal point can be changed by the user

## Output
- When the package is run first you can see the path that is generated for those points. Press any key to close the window.
- Then you can see the the turtlebot following the that was generated in the simulation.

