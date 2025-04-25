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
import cv2
import time
import tensorflow as tf
import numpy as np
import json
from pathlib import Path

from utilities.computer_vision_utils import *
# ====================================
# ====================================



# ===================================
# =========SUPPORT FUNCTIONS ========

# === ROBOT CONTROL FUNCTION ===
def robot_control(actual_state, x,y):
    # Implement servo control logic here
    print("Executed robot_control with: (%d,%d)" %(x,y))
    updated_state = actual_state
    return updated_state

# === ROBOT CONTROL FUNCTION ===
def robot_state_init():
    # Implement robot state initialization logic here
    print("Executed robot state init")
    state_t0 = 0
    return state_t0

# ===================================
# ===================================



# =============================
# === GLOBAL VAR DEFINITION ===
with open("offline_config.json", "r") as f:
    config = json.load(f)

VIDEO_PATH  = config["video_cfg"]["video_path"]      # Path to the local video
SKIP_FRAMES = config["video_cfg"]["skip_frames"]     # Number of frames to skip between each inference
MODEL_PATH  = config["video_cfg"]["model_path"]      # Path to the saved TensorFlow model

X_TARGET_SIZE = config["detection_cfg"]["x_resize"]             # target x size based on model training
Y_TARGET_SIZE = config["detection_cfg"]["y_resize"]             # target y size based on model training
CLASS_FILTER  = config["detection_cfg"]["class_filter"]         # class filter on classification output
CLASSIFIC_TH  = config["detection_cfg"]["classif_th"]           # treshold on classification score
FILTER_OBJ_NR = config["detection_cfg"]["filter_on_detect_nr"]  # filter on detected object number with higher score

classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]

# === MODEL LOADING ===
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

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video.")
        return

    frame_count = 0
    print("Starting video loop...")

    while cap.isOpened():
        ret, frame_gbr = cap.read()
        
        if not ret:
            break  # End of video

        if frame_count % (SKIP_FRAMES + 1) == 0:

            # store initial frame shape
            h, w, _ = frame_gbr.shape

            frame_rgb = frame_cv_to_tf_colours(frame_gbr)
            tf_frame_resized_uint8_batched, tf_frame_resized_float32 = prepro_frame(frame_rgb, x_size = X_TARGET_SIZE, y_size = Y_TARGET_SIZE)

            # get raw classification from mobilenet model
            boxes, scores, classes, num_detections = object_detection_fcn(model, tf_frame_resized_uint8_batched)
            
            # filter and order raw classification from mobilenet model based on cfg params
            classification_output = [boxes, scores, classes, num_detections]
            boxes_filt, scores_filt, classes_filt, num_detections_filt = filter_on_detection_nr(classification_output, classification_filter_param)
            
            # get single object to be tracked (the first one box is supposed to be the higher score for the class of interest)
            x_target, y_target = get_object_center_for_tracking(boxes_filt, h, w)

            robot_updated_state = robot_control(robot_actual_state, x_target, y_target)

        else:
            # keep the actual robot state
            robot_updated_state = robot_actual_state

        robot_actual_state = robot_updated_state
        frame_count += 1

        # # Slow down the loop slightly if needed to reduce CPU load
        # time.sleep(0.01) # check if needed

    cap.release()
    print("Video processing complete.")

if __name__ == "__main__":
    main()







