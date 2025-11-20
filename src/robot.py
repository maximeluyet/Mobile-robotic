# imports
import time

class Robot:
    ''' Main robots class'''

    def __init__(self, cfg, set_speed, stop, get_speed, get_ir, get_acc):
        self.cfg = cfg
        self.set_speed = set_speed
        self.stop = stop
        self.get_speed = get_speed
        self.get_ir = get_ir
        self.get_acc = get_acc

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
