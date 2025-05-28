# first attempt to main structure for classification management
#
# workflow:
# - setup video path (later on there will be a camera support)
# - cycle on video frame (with a customizable skippable frames nr)
# - for each not skipped frame, run computer vision model
# - get the coordinates wrt the center of the camera
# - call to empty functions (only signatures by far) for robot control and servo movement
#
# activate virtual env: source venv/bin/activate (in VS terminal)


# ====================================
# === IMPORT PY PKGS AND FUNCTIONS ===
from picamera2 import Picamera2
import cv2
import time
import tensorflow as tf
import numpy as np
import json
import datetime
from pathlib import Path
import time

from utilities.computer_vision_utils import *
# ====================================
# ====================================



# ===================================
# =========SUPPORT FUNCTIONS ========

# === ROBOT CONTROL FUNCTION ===
def robot_control(actual_state, x,y):
    # Implement servo control logic here
    print("-- executed robot_control with: (%d,%d)" %(x,y))
    updated_state = actual_state
    return updated_state

# === ROBOT CONTROL FUNCTION ===
def robot_state_init():
    # Implement robot state initialization logic here
    print("-- executed robot state init")
    state_t0 = 0
    return state_t0


# === HELPER FCN IN CASE OF OUTPUT WRITING ===
def draw_red_cross(frame, position, size=15, thickness=3):
    """
    Draws a red 'X' centered at 'position' on the frame.
    """
    x, y = position
    color = (0, 0, 255)  # Red color in BGR

    # Check and fix common OpenCV incompatibilities
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame is not a NumPy array.")
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)

    height, width = frame.shape[:2]

    # Clamp coordinates to avoid drawing outside image bounds
    x_min = max(x - size, 0)
    y_min = max(y - size, 0)
    x_max = min(x + size, width - 1)
    y_max = min(y + size, height - 1)

    cv2.line(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    cv2.line(frame, (x_min, y_max), (x_max, y_min), color, thickness)
    return frame
# ===================================
# ===================================



# =============================
# === GLOBAL VAR DEFINITION ===
MODEL_PATH  = "computer_vision/ssd_mobilenet_v2_320x320_coco17_tpu-8/TFLite/prepro_model_nodynamicinput/saved_model" # Path to the saved TensorFlow model
FLAG_OUTPUT_DISPL  = True   # flag for output video display enable

X_TARGET_SIZE = 320     # target x size based on model training
Y_TARGET_SIZE = 320     # target y size based on model training
CLASS_FILTER  = 37      # class filter on classification output
CLASSIFIC_TH  = 0.2     # treshold on classification score
FILTER_OBJ_NR = 1       # filter on detected object number with higher score

classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]


# === MODEL LOADING ===
print("-- computer vision model loading")
current_folder = Path(__file__).parent.resolve() # Get the directory of the current script
model_full_path = current_folder / MODEL_PATH
model = load_tf_saved_model(model_full_path)

# =============================
# =============================



# === MAIN LOOP ===
def main():

    # set robot init state
    robot_actual_state = robot_state_init()
    robot_updated_state = robot_actual_state

    # initialize the camera
    FPS = 30 # manual setting for expected pfs
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
                    main={"size": (X_TARGET_SIZE, Y_TARGET_SIZE)},
                    controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))}
                )            # to be considered: buffer_count=3 ?
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow the camera to warm up


    frame_count = 0
    print("Starting video loop...")

    while True:
        
        frame_rgb = picam2.capture_array()
        
        frame = frame_rgb
        if frame.shape[-1] == 4:
            frame = frame[..., :3]  # remove eventual transparency from picam2 module

        # store initial frame shape
        h, w, _ = frame.shape

        #frame_rgb = frame_cv_to_tf_colours(frame_gbr)
        tf_frame_resized_uint8_batched, tf_frame_resized_float32 = prepro_frame(frame, x_size = X_TARGET_SIZE, y_size = Y_TARGET_SIZE)

        # get raw classification from mobilenet model
        start_t = time.time()
        boxes, scores, classes, num_detections = object_detection_fcn(model, tf_frame_resized_uint8_batched)
        end_t = time.time()
        print(end_t-start_t)

        # filter and order raw classification from mobilenet model based on cfg params
        classification_output = [boxes, scores, classes, num_detections]
        boxes_filt, scores_filt, classes_filt, num_detections_filt = filter_on_detection_nr(classification_output, classification_filter_param)
            
        # get single object to be tracked (the first one box is supposed to be the higher score for the class of interest)
        x_target, y_target = get_object_center_for_tracking(boxes_filt, h, w)
        print((x_target, y_target))

        #################### FOR OUTPUT VIDEO SAVING PURPOSE ONLY #######################
        if FLAG_OUTPUT_DISPL:
            # === DRAWING RED 'X' IF TARGET VALID ===
            if (x_target != -1) and (y_target != -1):
                frame = draw_red_cross(frame, (x_target, y_target))
        
            cv2.imshow("Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # cv2 requires bgr frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #################################################################################

        if (x_target == -1 & y_target == -1):
            # keep the actual robot state and do not move
            robot_updated_state = robot_actual_state

        else:
            # update robot state after movement
            robot_updated_state = robot_control(robot_actual_state, x_target, y_target)

        # robot_actual_state = robot_updated_state
        frame_count += 1

    # Clean up
    picam2.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







