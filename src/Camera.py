import cv2
import time
from picamera2 import Picamera2


class Camera:
    def __init__(self, width=320, height=320, fps=30, focal_length=800):

        # Camera parameters
        self.width = width
        self.height = height
        self.focal_length = focal_length

        self._camera = self._init_camera(fps)

    def _init_camera(self, fps):
        picam2 = Picamera2()

        config = picam2.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888"
            },
            controls={
                "FrameDurationLimits": (
                    int(1e6 / fps),
                    int(1e6 / fps)
                )
            }
        )

        picam2.configure(config)
        picam2.start()

        time.sleep(2)  # camera warm-up

        return picam2

    def get_camera_frame(self):
        # Always copy to avoid buffer issues
        frame = self._camera.capture_array("main").copy()

        # Safety: remove alpha if present (rare in this config)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        return frame

    @staticmethod
    def frame_cv_to_tf_colours(frame_bgr):
        # OpenCV BGR -> RGB (for ML models)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
