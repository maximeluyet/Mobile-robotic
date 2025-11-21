import numpy as np

# Module constants:
STATE_DIM = 5
P_INIT = np.zeros((STATE_DIM, STATE_DIM)) # TODO initial covariance matrix
R_CAM = np.zeros(STATE_DIM) # TODO
R_NO_CAM = np.eye(2) # Measurement noise covariance without camera
Q = np.diag([0.1, 0.1, 0.05, 0.1, 0.05]) # Process noise covariance
H = np.array([[0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])  # Measurement matrix for no camera update

WHEEL_BASE = 94# mm


class KalmanFilter:
    ''' Extended Kalman Filter implementation for state estimation. '''
    
    def __init__(self):
        
        self.initialized = False

        pass

    def init_state(self, camera_pose):
        px = camera_pose[0]
        py = camera_pose[1]
        theta = camera_pose[2]

        # Set state with zero velocity
        self.x = np.array([px, py, theta, 0, 0])
        
        # Initialize covariance
        self.P = P_INIT

        self.initialized = True

    def set_covered(self, covered):
        ''' Adjust measurement noise based on whether the camera is covered or not '''
        if covered:
            self.R *= 0.5
        else:
            self.R *= 2.0
    
    def predict(self, left_speed_c, right_speed_c, dt):
        ''' Predict current state using motion model with last state and control inputs '''

        # Get linear and angular velocities from wheel speeds
        v, w = self._velocity_from_wheels(left_speed_c, right_speed_c)
        print(f"Predict step with v: {v}, w: {w}, dt: {dt}")

        # Update state based on motion model
        self.x[0] = self.x[0] + v * np.cos(self.x[2]) * dt
        self.x[1] = self.x[1] + v * np.sin(self.x[2]) * dt
        self.x[2] = self.x[2] + w * dt
        self.x[3] = v
        self.x[4] = w

        # Linearize motion model
        F_k = self._linearize_motion(self.x, dt)

        # Update covariance
        self.P = F_k @ self.P @ F_k.T + Q
        

    def update(self, camera_pose_m, left_speed_m, right_speed_m):
        ''' Update the state using the camera pose and wheel speeds measurements '''

        # Get linear and angular velocities from wheel speeds
        v_m, w_m = self._velocity_from_wheels(left_speed_m, right_speed_m)

        z = np.concatenate(camera_pose_m, [v_m, w_m])

        # Calculate innovation
        y = z - self.x

        # Update covariance and get kalman gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # Posterior belief
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K) @ self.P

        return self.x, self.P
    
    def update_no_camera(self, left_speed_m, right_speed_m):
        ''' Update the state using only wheel speeds measurements '''

        # Get linear and angular velocities from wheel speeds
        v_m, w_m = self._velocity_from_wheels(left_speed_m, right_speed_m)

        z = np.array([v_m, w_m])

        # Calculate innovation
        y = z - H @ self.x

        # Update covariance and get kalman gain
        S = H @ self.P @ H.T + R_NO_CAM
        K = self.P @ H.T @ np.linalg.inv(S)

        # Posterior belief
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ H) @ self.P

        return self.x, self.P

    
    def get_state_cov(self):
        ''' Return current state and covariance '''
        
        return self.x, self.P

    
    @staticmethod
    def _linearize_motion(x, dt):
        ''' Linearize the motion model around the current state x '''
        
        return np.array([[1, 0, -x[3]*np.sin(x[2]), np.cos(x[2]), 0],
                         [0, 1,  x[3]*np.cos(x[2]), np.sin(x[2]), 0],
                         [0, 0, 1, 0, dt],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1]])
    
    @staticmethod
    def _velocity_from_wheels(left_speed, right_speed):
        ''' Convert wheel speeds to linear and angular velocities '''
        
        v = (left_speed + right_speed) / 2.0
        w = (right_speed - left_speed) / WHEEL_BASE
        
        return v, w