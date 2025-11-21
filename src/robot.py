# imports
import time
import numpy as np
from filtering import KalmanFilter

## Constants
MM_SEC_TO_THYMIO = 33.21 / 100  # TODO: update this
TIME_STEP = 0.2  # seconds

class Robot:
    ''' Main robots class'''

    def __init__(self, set_speed, stop, get_speed, get_ir, get_acc):
        self.set_speed = set_speed
        self.stop = stop
        self.get_speed = get_speed
        self.get_ir = get_ir
        self.get_acc = get_acc

        self.ekf = KalmanFilter()

        pass

    def run(self):
        ''' Main state machine of the robot'''
        
        # Example rotate both ways
        self.set_speed(50, -50)
        time.sleep(0.5)
        self.set_speed(-50, 50)
        time.sleep(0.5)
        self.stop()     

        # TODO: implement robot behavior
        pass
        

    def try_kalman(self):
        ''' Drive straight at given speed for given duration '''

        speeds = []
        model_positions = []
        model_Ps = []

        # initialize kalman filter
        self.ekf.init_state(np.zeros(3))
        
        # go linear for 3 sec
        self.set_speed(200, 200)
        
        start_time = time.time()

        while (time.time() - start_time) < 3.0:
            left_speed, right_speed = self.get_speed()
            speeds.append([left_speed, right_speed])

            v_l = self.thymio_to_mm_sec(left_speed)
            v_r = self.thymio_to_mm_sec(right_speed)

            # predict step
            self.ekf.predict(v_l, v_r, TIME_STEP)
            model_pos, model_P = self.ekf.get_state_cov()

            model_positions.append(model_pos.copy())
            model_Ps.append(model_P.copy())

            time.sleep(TIME_STEP)
        self.stop()

        # turn for 1 sec
        self.set_speed(-200, 200)

        start_time = time.time()

        while (time.time() - start_time) < 1.0:
            left_speed, right_speed = self.get_speed()
            speeds.append([left_speed, right_speed])

            v_l = self.thymio_to_mm_sec(left_speed)
            v_r = self.thymio_to_mm_sec(right_speed)

            # predict step
            self.ekf.predict(v_l, v_r, TIME_STEP)
            model_pos, model_P = self.ekf.get_state_cov()

            model_positions.append(model_pos.copy())
            model_Ps.append(model_P.copy())

            time.sleep(TIME_STEP)
        self.stop()

        # go straight back for 3 sec
        self.set_speed(250, 250)
        v_rl = self.thymio_to_mm_sec(100)
        
        start_time = time.time()

        while (time.time() - start_time) < 3.0:
            left_speed, right_speed = self.get_speed()
            speeds.append([left_speed, right_speed])

            v_l = self.thymio_to_mm_sec(left_speed)
            v_r = self.thymio_to_mm_sec(right_speed)

            # predict step
            self.ekf.predict(v_l, v_r, TIME_STEP)
            model_pos, model_P = self.ekf.get_state_cov()

            model_positions.append(model_pos.copy())
            model_Ps.append(model_P.copy())

            time.sleep(TIME_STEP)
        self.stop()

        return speeds, model_positions, model_Ps
    
    @staticmethod
    def mm_sec_to_thymio(speed_mm_sec):
        return int(speed_mm_sec / MM_SEC_TO_THYMIO)
    
    @staticmethod
    def thymio_to_mm_sec(speed_thymio):
        return speed_thymio * MM_SEC_TO_THYMIO
    