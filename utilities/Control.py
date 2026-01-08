import numpy as np
from pathlib import Path
import time

class Control:
    def __init__(self, CV_MODEL_PRJ_PATH = "",
                 X_TARGET_SIZE = 320, Y_TARGET_SIZE = 320,
                 CLASS_FILTER = 37, CLASSIFIC_TH = 0.2, FILTER_OBJ_NR = 1):

        """
        Tracker contructor function

        # -- default params for sport ball deteciton on mobile net sdd cv model
        X_TARGET_SIZE = 320     # target x size based on model training - mobile net training x size
        Y_TARGET_SIZE = 320     # target y size based on model training - mobile net training y size
        CLASS_FILTER  = 37      # class filter on classification output
        CLASSIFIC_TH  = 0.2     # treshold on classification score
        FILTER_OBJ_NR = 1       # filter on detected object number with higher score
        """

        # Tracker actual state
        self.prev_target_coord = (-1, -1)
        self.servo1_deg = 0
        self.servo2_deg = 0
        self.is_servo_moving = False
        

    #########################
    # Control methods

    # === LOAD MODEL ===
    def __get_cv_model(self, model_path_in_prj):
        print("-- computer vision model loading")
        current_folder = Path(__file__).parent.resolve()
        model_full_path = current_folder.parent / model_path_in_prj

        return tf.saved_model.load(str(model_full_path))

    # === INITIALIZE CAMERA ===
    def __init_Pi_camera(self, FPS = 30):

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
                        main={"size": (self.detection.__x_target_size, self.detection.__y_target_size)},
                        controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))}
                    )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        return picam2

    def get_camera_frame(self):
        frame_rgb = self.detection.__camera.capture_array()
        
        frame = frame_rgb
        if frame.shape[-1] == 4:
            frame = frame[..., :3]  # remove eventual transparency from picam2 module

        # store initial frame shape
        self.detection.frame_orig_h, self.detection.frame_orig_w, _ = frame.shape

        return frame


    #########################
    """
    #############################
    # Tracker and control methods

    # === ROBOT CONTROL FUNCTION ===
    def robot_control(self, actual_state, x,y):
        # Implement servo control logic here
        print("Executed robot_control with: (%d,%d)" %(x,y))
        self.track_state.updated_state = self.track_state.actual_state
        return self.track_state.updated_state

    # === ROBOT CONTROL FUNCTION ===
    def robot_state_init(self):
        # Implement robot state initialization logic here
        print("Executed robot state init")
        self.track_state.state_t0 = []
        return self.track_state.state_t0


    #############################
        
    """
