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
        # white background
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # update ball position (no bouncing)
        self.ball_pos += self.ball_vel

        # camera rotation → image-plane shift
        shift_x = self.focal_length * self.rot_z
        shift_y = self.focal_length * self.rot_x

        # apparent position in image
        self.ball_virtual_pos[0] = int(self.ball_virtual_pos[0] - shift_x + self.ball_vel[0])
        self.ball_virtual_pos[1] = int(self.ball_virtual_pos[1] - shift_y + self.ball_vel[1])

        # draw ball only if inside image
        if 0 <= self.ball_virtual_pos[0] < self.width and 0 <= self.ball_virtual_pos[1] < self.height:
            cv2.circle(
                frame,
                (int(self.ball_virtual_pos[0]), int(self.ball_virtual_pos[1])),
                self.ball_radius,
                self.ball_color,
                -1
            )

        return frame