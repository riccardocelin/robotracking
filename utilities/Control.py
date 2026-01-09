import numpy as np
import RPi.GPIO as GPIO
from time import sleep

class Control:
    def __init__(self, detector, Control_type="P"):

        """
        Control contructor function

        """

        # Control actual state
        self.detector_obj = detector
        self.control_type = Control_type
        self.prev_target_coord = (-1, -1)
        self.servo1_deg = 0
        self.servo2_deg = 0
        self.is_servo_moving = False
        

    #########################
    # Control methods
    # === ROBOT CONTROL FUNCTION ===
    def robot_control(self):
        # Implement servo control logic here
        self.track_state.updated_state = self.track_state.actual_state
        return self.track_state.updated_state


