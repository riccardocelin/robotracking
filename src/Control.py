class Control:
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
        self.prev_uz = 0
        self.prev_ux = 0
        

    #########################
    # Control methods
    # === ROBOT CONTROL FUNCTION ===
    def actuation_init(self):
        # Implement servo control logic here

        if self.control_type == "P":
            theta_z_target = (self.x_target_px - self.x_center)/self.focal_length
            theta_x_target = (self.y_target_px - self.y_center)/self.focal_length
            err_z = theta_z_target
            err_x = theta_x_target
            u_z = err_z * self.Proportional_gain  # Proportional gain for Z axis
            u_x = err_x * self.Proportional_gain  # Proportional gain for X axis

        
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

    def update_target_coords(self, x_target, y_target):
        # Update target coordinates in the detector object
        self.x_target_px = x_target
        self.y_target_px = y_target

    def set_control_action(self, u_z, u_x):

        #map u_z and u_x according to 0->7, -10->1, 10->14
        if u_z>0:
            u_z = 35*u_z + 7
        else:
            u_z = 30*u_z+7
            
        if u_x>0:
            u_x = 35*u_x + 7
        else:
            u_x = 30*u_x+7
            
        u_x = min(max(1, u_x), 100)
        u_z = min(max(1, u_z), 100)
        
        self.prev_ux = u_x
        self.prev_uz = u_z
        
        return u_z,u_x 
        




