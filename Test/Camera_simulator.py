import numpy as np
import cv2

class CameraSimulator:
    def __init__(self, width=320, height=320):
        self.width = width
        self.height = height

        # Ball parameters
        self.ball_radius = 20
        self.ball_color = (0, 140, 255)
        self.ball_pos = np.array([width // 4, height // 2], dtype=float)
        self.ball_vel = np.array([1.0, 0.5])  # (pixels/frame)

    def get_camera_frame(self):
        # white background
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # update ball position
        self.ball_pos += self.ball_vel

        # bounce on borders
        if self.ball_pos[0] <= self.ball_radius or self.ball_pos[0] >= self.width - self.ball_radius:
            self.ball_vel[0] *= -1
        if self.ball_pos[1] <= self.ball_radius or self.ball_pos[1] >= self.height - self.ball_radius:
            self.ball_vel[1] *= -1

        # draw ball
        cv2.circle(
            frame,
            center=tuple(self.ball_pos.astype(int)),
            radius=self.ball_radius,
            color=self.ball_color,
            thickness=-1
        )

        return frame