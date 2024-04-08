#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import tty
import termios
from pynput import keyboard

# Define key codes
LIN_VEL_STEP_SIZE = 0.1
ANG_VEL_STEP_SIZE = 0.1

class KeyboardControlNode(Node):

    def __init__(self):
        super().__init__('keyboard_control_node')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.settings = termios.tcgetattr(sys.stdin)

    def getKey(self):
        """Get the key that is pressed"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_keyboard_control(self):
        """Run the keyboard control node"""
        
        self.msg = """
        Control Your Car!
        ---------------------------
        Moving around:
            w
        a    s    d

        q : force stop

        Esc to quit
        """

        self.get_logger().info(self.msg)
        velocity_message = Twist()
        linear_vel=0.0
        angular_vel=0.0


        while True:
            key = self.getKey()
            if key is not None:
                if key == '\x1b':  # Escape key
                    break
                elif key == 'q':  # Quit
                    linear_vel=0.0
                    angular_vel=0.0
                elif key == 'w':  # Forward
                    linear_vel += LIN_VEL_STEP_SIZE
                elif key == 's':  # Reverse
                    linear_vel -= LIN_VEL_STEP_SIZE
                elif key == 'd':  # Right
                    angular_vel -= ANG_VEL_STEP_SIZE
                elif key == 'a':  # Left
                    angular_vel += ANG_VEL_STEP_SIZE


                if angular_vel>1.0:
                        angular_vel=1.0
                if angular_vel<-1.0:
                    angular_vel=-1.0

                print("Steer Angle",angular_vel)
                print("Linear Velocity",linear_vel)
                
                # Publish the twist message
                velocity_message.linear.x = linear_vel
                velocity_message.angular.z = angular_vel
                
                self.cmd_vel_pub.publish(velocity_message)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    node.run_keyboard_control()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()