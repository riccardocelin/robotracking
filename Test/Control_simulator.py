import numpy as np
from time import sleep

class ControlSimulator:
    def __init__(self, Control_type="P", focal_length=None, x_center=160, y_center=160):

        """
        Control contructor function
        """

        # Control actual state
        self.control_type = Control_type
        self.Proportional_gain = 2
        self.x_target_px = -1
        self.y_target_px = -1
        self.focal_length = focal_length  # pixels
        self.x_center = x_center
        self.y_center = y_center
        
    #########################
    # Control methods
    # === ROBOT CONTROL FUNCTION ===
    def update_target_coords(self, x_target, y_target):
        # Update target coordinates in the detector object
        self.x_target_px = x_target
        self.y_target_px = y_target

    def robot_control(self):
        # Implement servo control logic here

        if self.control_type == "P":
            theta_z_target = (self.x_target_px - self.x_center)/self.focal_length
            theta_x_target = (self.y_target_px - self.y_center)/self.focal_length
            err_z = theta_z_target
            err_x = theta_x_target
            u_z = err_z * self.Proportional_gain  # Proportional gain for Z axis
            u_x = err_x * self.Proportional_gain  # Proportional gain for X axis
        return u_z, u_x