import numpy as np
from dataclasses import dataclass
import time

# Module constants:
STATE_DIM = 3

# Process noise covariance matrix
Q = np.diag([0.002, 0.001, 0.05]) # TODO: analyze & tune more

# Measurement noise covariance matrix (camera noise)
R = np.eye(STATE_DIM) * 0.01 # TODO: analyze & tune more
# R[2, 2] = 1.2 # TODO: maybe add this

WHEEL_BASE = 94 # mm

class ExtendedKalmanFilter:
    ''' Extended Kalman Filter implementation for state estimation. '''
    
    def __init__(self):
        
        self.initialized = False

        self.x = np.zeros(STATE_DIM)
        self.P = np.zeros((STATE_DIM, STATE_DIM))

        self.last_prediction_time = 0.0

        # Initialize history for plotting
        self.pose_hist = []
        self.cov_hist = []
        self.meas_hist = []
    

    def init_state(self, camera_pose_m):

        # set time reference
        self.last_prediction_time = time.time()

        # Set state with zero velocity
        self.x = camera_pose_m.copy()
        
        # Initialize covariance
        self.P = R.copy()

        self.initialized = True
    
    
    def predict(self, right_speed_m, left_speed_m):
        ''' Predict current state using motion model with last state and control inputs '''

        # Get time delta
        current_time = time.time()
        dt = current_time - self.last_prediction_time
        self.last_prediction_time = current_time

        # Get linear and angular velocities from wheel speeds
        v_lin = (right_speed_m + left_speed_m) / 2.0
        v_ang = (right_speed_m - left_speed_m) / WHEEL_BASE

        # Linearize motion model around previous state
        F_x = self._linearize_motion(self.x, right_speed_m, left_speed_m, dt)

        # Update state based on motion model
        self.x[0] = self.x[0] + v_lin * np.cos(self.x[2]) * dt
        self.x[1] = self.x[1] + v_lin * np.sin(self.x[2]) * dt
        self.x[2] = self.x[2] + v_ang * dt

        # Wrap angle to [-pi, pi]
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Update covariance
        self.P = F_x @ self.P @ F_x.T + Q

        return self.x
        

    def update(self, camera_pose_m, save_history=False):
        ''' Update the state using the camera pose '''

        # Calculate innovation
        y = camera_pose_m - self.x

        # Get Kalman Gain
        K = self.P @ np.linalg.inv(self.P + R)

        # Posterior belief
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K) @ self.P

        if save_history:
            self.meas_hist.append(camera_pose_m.copy())

    
    def get_state_cov(self, save_history=False):
        ''' Return current state and covariance '''

        if save_history:
            self.pose_hist.append(self.x.copy())
            self.cov_hist.append(self.P.copy())
        
        return self.x, self.P


    def get_history(self):
        # Return the ekf history

        return self.pose_hist, self.cov_hist, self.meas_hist
    

    def clear_history(self):
        # Clear the ekf history

        self.pose_hist = []
        self.cov_hist = []
        self.meas_hist = []
    

    @staticmethod
    def _linearize_motion(x, v_r, v_l, dt):
        ''' Linearize the motion model around the state x '''

        v_lin = (v_r + v_l) / 2.0
        theta = x[2]

        F_x = np.eye(STATE_DIM)
        F_x[0, 2] = - v_lin * dt * np.sin(theta)
        F_x[1, 2] = v_lin * dt * np.cos(theta)
        
        return F_x
    

@dataclass
class CoordinateTransform2D:
    px_per_mm: float
    map_height_mm: float

    def vision_to_ekf(self, pose):
        x, y, theta = pose

        x_e = x / self.px_per_mm
        y_e = self.map_height_mm - (y / self.px_per_mm)
        theta_e = -theta

        return np.array([x_e, y_e, theta_e])
    
    def ekf_to_vision(self, pose):
        x_e, y_e, theta_e = pose

        x = x_e * self.px_per_mm
        y = (self.map_height_mm - y_e) * self.px_per_mm
        theta = -theta_e

        return np.array([x, y, theta])


@dataclass
class WheelSpeedTransform:
    thymio_to_mm_sec_right: float
    thymio_to_mm_sec_left: float

    def thymio_to_mmsec(self, left_speed, right_speed):

        left_mm_sec = left_speed / self.thymio_to_mm_sec_left
        right_mm_sec = right_speed / self.thymio_to_mm_sec_right

        return left_mm_sec, right_mm_sec
    
    def mmsec_to_thymio(self, left_speed, right_speed):

        left_thymio = left_speed * self.thymio_to_mm_sec_left
        right_thymio = right_speed * self.thymio_to_mm_sec_right

        return left_thymio, right_thymio