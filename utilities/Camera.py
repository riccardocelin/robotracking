import cv2
import numpy as np
import time
from picamera2 import Picamera2


class Camera:
    def __init__(self, X_TARGET_SIZE = 320, Y_TARGET_SIZE = 320):

        """
        Tracker contructor function

        # -- default params for sport ball deteciton on mobile net sdd cv model
        X_TARGET_SIZE = 320     # target x size based on model training - mobile net training x size
        Y_TARGET_SIZE = 320     # target y size based on model training - mobile net training y size
        CLASS_FILTER  = 37      # class filter on classification output
        CLASSIFIC_TH  = 0.2     # treshold on classification score
        FILTER_OBJ_NR = 1       # filter on detected object number with higher score
        """

        # detection settings
        self.__x_target_size = X_TARGET_SIZE
        self.__y_target_size = Y_TARGET_SIZE
        self.__camera = self.__init_Pi_camera()

    #########################
    # Computer vision methods

    # === INITIALIZE CAMERA ===
    def __init_Pi_camera(self, FPS = 30):

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
                        main={"size": (self.__x_target_size, self.__y_target_size)},
                        controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))}
                    )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        return picam2

    def get_camera_frame(self):
        frame_rgb = self.__camera.capture_array()
        
        frame = frame_rgb
        if frame.shape[-1] == 4:
            frame = frame[..., :3]  # remove eventual transparency from picam2 module

        return frame
        
    # === FRAME PREPROCESSING ===
    def frame_cv_to_tf_colours(self, frame_bgr):
        # Convert from BGR (OpenCV) to RGB (TensorFlow)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb
