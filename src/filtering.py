import numpy as np

# Module constants:
STATE_DIM = 3
INPUT_DIM = 2

# Initial state covariance matrix
P_INIT = np.zeros((STATE_DIM, STATE_DIM)) # TODO: currently zero, get unit

# Process noise covariance matrix (wheel speed noise)
Q = np.eye(INPUT_DIM)*3 # TODO: currently identity -> measure, get unit
# Q[0,0] = 0.5

# Measurement noise covariance matrix (camera noise)
R = np.zeros(STATE_DIM) # TODO: currently identity -> measure, get unit
# R = np.eye(STATE_DIM) * 0.1
# R[2, 2] = 0

WHEEL_BASE = 94 # mm # TODO: correct maybe


class ExtendedKalmanFilter:
    ''' Extended Kalman Filter implementation for state estimation. '''
    
    def __init__(self, keepHist=False):
        
        self.initialized = False

    
    def init_state(self, camera_pose_m):

        # Set state with zero velocity
        self.x = camera_pose_m.copy()
        
        # Initialize covariance
        self.P = P_INIT

        self.initialized = True
    
    
    def predict(self, right_speed_m, left_speed_m, dt, verbose=False):
        ''' Predict current state using motion model with last state and control inputs '''

        # Get linear and angular velocities from wheel speeds
        v_lin = (right_speed_m + left_speed_m) / 2.0
        v_ang = (right_speed_m - left_speed_m) / WHEEL_BASE
        
        if verbose:
            print(f"Predict step with v_lin: {v_lin}, v_ang: {v_ang}, dt: {dt}")

        # Update state based on motion model
        self.x[0] = self.x[0] + v_lin * np.cos(self.x[2]) * dt
        self.x[1] = self.x[1] + v_lin * np.sin(self.x[2]) * dt
        self.x[2] = self.x[2] + v_ang * dt

        # Linearize motion model
        F_x, F_u = self._linearize_motion(self.x, right_speed_m, left_speed_m, dt)

        # Update covariance
        self.P = F_x @ self.P @ F_x.T + F_u @ Q @ F_u.T
        

    def update(self, camera_pose_m, verbose=False):
        ''' Update the state using the camera pose '''

        # Calculate innovation
        y = camera_pose_m - self.x

        # Get Kalman Gain
        K = self.P @ np.linalg.inv(self.P + R)

        # Posterior belief
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K) @ self.P

        return self.x, self.P

    
    def get_state_cov(self):
        ''' Return current state and covariance '''
        
        return self.x, self.P

    
    @staticmethod
    def _linearize_motion(x, v_r, v_l, dt):
        ''' Linearize the motion model around the state x '''

        v_lin = (v_r + v_l) / 2.0
        theta = x[2]

        F_x = np.eye(STATE_DIM)
        F_x[0, 2] = - v_lin * dt * np.sin(theta)
        F_x[1, 2] = v_lin * dt * np.cos(theta)

        F_u = np.zeros((STATE_DIM, INPUT_DIM))
        F_u[0, :] = np.cos(theta) * dt / 2
        F_u[1, :] = np.sin(theta) * dt / 2
        F_u[2, 1] = dt / WHEEL_BASE
        F_u[2, 0] = -dt / WHEEL_BASE
        
        return F_x, F_u
    