import numpy as np
import cv2

class CameraSimulator:
    def __init__(self, width=320, height=320, focal_length=800):
        self.width = width
        self.height = height
        self.focal_length = focal_length  # pixels

        # Ball parameters
        self.ball_radius = 20
        self.ball_color = (0, 140, 255)
        self.ball_pos = np.array([width // 3, height // 3], dtype=float)
        self.ball_virtual_pos = self.ball_pos.copy()
        self.ball_vel = np.array([1.0, 0.5])+np.random.randn(2) * 0.5  # random velocity

        # Camera rotation state
        self.rot_z = 0.0
        self.rot_x = 0.0

    def set_camera_rotation(self, rot_z=0.0, rot_x=0.0):
        """
        rot_z : rotation around z axis in radians
        rot_x : rotation around X axis in radians
        """
        self.rot_z = rot_z
        self.rot_x = rot_x

    def get_camera_frame(self):

        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # update real position
        self.ball_pos += self.ball_vel

        # camera rotation (ASSUMING radians)
        shift_x = self.focal_length * self.rot_z
        shift_y = self.focal_length * self.rot_x

        # IMPORTANT: do NOT accumulate virtual position
        virtual_x = self.ball_pos[0] - shift_x
        virtual_y = self.ball_pos[1] - shift_y

        if 0 <= virtual_x < self.width and 0 <= virtual_y < self.height:
            cv2.circle(
                frame,
                (int(virtual_x), int(virtual_y)),
                self.ball_radius,
                self.ball_color,
                -1
            )

        return frame