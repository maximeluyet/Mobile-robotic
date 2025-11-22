# imports
import numpy as np


class LocalNavigation:

    def __init__(self):
        # Weights for a direction vectors in opposition to the sensors
        self.w = [[10,-30]/np.linalg.norm([10,-30]),      # sensor x1
            [5,-20]/np.linalg.norm([5,-20]),         # sensor x2
            [0,-10]/np.linalg.norm([0,-10]),         # sensor x3
            [-5, -20]/np.linalg.norm([-5, -20]),     # sensor x4
            [-10, -30]/np.linalg.norm([-10, -30]),   # sensor x5
            [15, 15]/np.linalg.norm([15, 15]),       # sensor x6
            [-15, 15]/np.linalg.norm([-15, 15])]     # sensor x7
        
        # Scale factors for sensors and constant factor
        self.sensor_scale = 200
        self.constant_scale = 20

        self.x = [0,0,0,0,0,0,0,0,0]

        self.direction_repulsion = [0,0]


    def compute(self, sensor_data, state):   # sensor_data = proximal_horizontal     
        if state != 0:
            # Memory
            self.x[7] = self.direction_repulsion[0]//10
            self.x[8] = self.direction_repulsion[1]//10

            for i in range(7):
                # Get and scale inputs
                self.x[i] = sensor_data[i] // self.sensor_scale

            self.direction_repulsion = [0,0]
            
            for i in range(len(self.x)):
                # Compute outputs of neurons and set motor powers
                self.direction_repulsion[0] += self.x[i, 0] * self.w[i]
                self.direction_repulsion[1] += self.x[i, 1] * self.w[i]
        else:
            # In case we would like to stop the robot
            self.direction_repulsion = [0,0]
        return self.direction_repulsion

        