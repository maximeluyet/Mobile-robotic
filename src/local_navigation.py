# imports
import numpy as np


class LocalNavigation:

    def __init__(self):
        # Weights for a direction vectors in opposition to the sensors
        # Use numpy arrays and normalize each vector
        self.w = [
            np.array([10, -20]) / np.linalg.norm([10, -20]),  # sensor x1
            np.array([5, -20]) / np.linalg.norm([5, -20]),    # sensor x2
            np.array([0, -10]) / np.linalg.norm([0, -10]),    # sensor x3
            np.array([-5, -20]) / np.linalg.norm([-5, -20]),  # sensor x4
            np.array([-10, -20]) / np.linalg.norm([-10, -20]),# sensor x5
            np.array([15, 15]) / np.linalg.norm([15, 15]),    # sensor x6
            np.array([-15, 15]) / np.linalg.norm([-15, 15]),  # sensor x7
            np.array([0, 0]) / np.linalg.norm([1, 0]),        # memory x8
            np.array([0, 0]) / np.linalg.norm([0, 1])         # memory x9
        ]
        
        # Scale factors for sensors and constant factor
        self.sensor_scale = 200
        self.constant_scale = 20

        # internal neuron values and direction repulsion as numpy arrays
        self.x = np.zeros(9, dtype=float)
        self.direction_repulsion = np.zeros(2, dtype=float)


    def compute(self, sensor_data, state):   # sensor_data = proximal_horizontal
        if len(sensor_data) < 7:
            raise ValueError("sensor_data must contain at least 7 elements.")
        else:
            self.x[:7] = sensor_data.copy()
        if state != 0:
            # Memory
            # store a coarse memory of previous direction
            self.x[7] = self.direction_repulsion[0] / 10.0
            self.x[8] = self.direction_repulsion[1] / 10.0

            # Safely read up to 7 sensor inputs (guard against short lists)
            max_inputs = min(7, len(sensor_data))
            for i in range(max_inputs):
                # Get and scale inputs (use float division)
                try:
                    self.x[i] = float(sensor_data[i]) / float(self.sensor_scale)
                except Exception:
                    # if sensor_data contains unexpected values, set to zero
                    self.x[i] = 0.0

            # reset repulsion and accumulate weighted directions
            self.direction_repulsion[0] = 0.0
            self.direction_repulsion[1] = 0.0
            for i, w_vec in enumerate(self.w):
                # protect against mismatch in lengths (use only available x entries)
                if i >= len(self.x):
                    print("Warning: mismatch in lengths of weights and internal states.")
                    break
                #print("weights vector:", w_vec, " internal state x[", i, "] =", self.x[i])
                self.direction_repulsion += self.x[i] * w_vec
        else: # State 0: no movement
            # In case we would like to stop the robot
            # assign element-wise to be compatible with list or numpy array
            self.direction_repulsion[0] = 0.0
            self.direction_repulsion[1] = 0.0

        # return plain python list for compatibility
        return [float(self.direction_repulsion[0]), float(self.direction_repulsion[1])]

        