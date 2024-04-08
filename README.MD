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


## Steps to run Part 2

