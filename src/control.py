# imports
import numpy as np
import math


# Tunables
k_rho, k_alpha, k_beta = 0.6, 1.2, -0.3
l = 0.047  # half axle length [m](adjust%20for%20Thymio%20~%2094%20mm%20axle)
r = 0.022  # wheel radius [m](adjust%20for%20Thymio%20wheel)
scale = 0.01  # maps rad/s to motor target units (tune)
rho_stop = 0.03  # [m]

class Control:
    ''' Main control class'''

    def __init__(self):
        pass

    # Helpers
    def wrapToPi(self, a):
        while a > math.pi: 
            a -= 2*math.pi
        while a < -math.pi: 
            a += 2*math.pi
        return a
    
    def control(self,robot_pose, goal_pose):
        ''' Astolfi Controller for position control'''
        dx = goal_pose[0] - robot_pose[0]
        dy = goal_pose[1] - robot_pose[1]
        rho = math.hypot(dx, dy)
        if rho < rho_stop:
            # Final orientation alignment
            alpha = 0.0
            beta  = self.wrapToPi(goal_pose[2] - robot_pose[2])
        else:
            alpha = self.wrapToPi(math.atan2(dy, dx) - robot_pose[2])
            beta  = self.wrapToPi(goal_pose[2] - robot_pose[2] - alpha)

        # Control law
        v = k_rho * rho
        print("v =", v)
        omega = k_alpha * alpha + k_beta * beta
        print("omega =", omega)

        # Map to wheels
        wL = (v - l*omega)/r
        wR = (v + l*omega)/r

        # Convert to Thymio units and saturate
        left_wheel_speed = int(max(min(scale * wL, 400), -400))
        right_wheel_speed = int(max(min(scale * wR, 400), -400))

        return (left_wheel_speed, right_wheel_speed)
