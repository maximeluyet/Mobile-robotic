import numpy as np
import math



class Control:
    ''' Main control class'''

    def __init__(self):
        self.rho_stop = 0.02  # [m]
        self.k_rho = 1.5
        self.k_alpha = 2.2
        self.k_beta = -1.0
        self.l = 0.047  # half axle length [m]
        self.r = 0.022  # wheel radius [m]
        self.scale = 20  # maps rad/s to motor target units (tune)

    # Helpers
    def wrapToPi(self, a): # wrap angle to [-pi, pi]
        while a > math.pi: 
            a -= 2*math.pi
        while a < -math.pi: 
            a += 2*math.pi
        return a
    
    def control(self, robot_pose, goal_pose):
        ''' 
        Astolfi Controller (Forward Only)
        robot_pose: [x, y, theta]
        goal_pose:  [x, y, theta]
        '''
        # 1. Calculate Error Vector
        dx = goal_pose[0] - robot_pose[0]
        dy = goal_pose[1] - robot_pose[1]
        rho = math.hypot(dx, dy)
        theta = robot_pose[2]
        alpha = self.wrapToPi(math.atan2(dy, dx) - theta)
        beta  = self.wrapToPi(goal_pose[2] - theta - alpha)

        # Control law
        v = self.k_rho * rho
        # print("v =", v)
        omega = self.k_alpha * alpha + self.k_beta * beta
        # print("omega =", omega)

        # Map to wheels
        wL = (v - self.l*omega)/self.r
        wR = (v + self.l*omega)/self.r

        # Convert to Thymio units and saturate
        left_wheel_speed = int(max(min(self.scale * wL, 400), -400))
        right_wheel_speed = int(max(min(self.scale * wR, 400), -400))
        return (left_wheel_speed, right_wheel_speed)