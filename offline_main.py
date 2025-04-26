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
import datetime
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


# === HELPER FCN IN CASE OF OUTPUT WRITING ===
def draw_red_cross(frame, position, size=15, thickness=3):
    """
    Draws a red 'X' centered at 'position' on the frame.
    """
    x, y = position
    color = (0, 0, 255)  # Red color in BGR
    cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness)
    return frame
# ===================================
# ===================================



# =============================
# === GLOBAL VAR DEFINITION ===
with open("offline_config.json", "r") as f:
    config = json.load(f)

VIDEO_PATH  = config["video_cfg"]["video_path"]      # Path to the local video
SKIP_FRAMES = config["video_cfg"]["skip_frames"]     # Number of frames to skip between each inference
MODEL_PATH  = config["video_cfg"]["model_path"]      # Path to the saved TensorFlow model
FLAG_OUTPUT_SAVE  = config["video_cfg"]["enable_output_video_saving"]   # flag for output video saving enable
OUTPUT_VIDEO_PATH = config["video_cfg"]["output_video_path"]            # output video path

X_TARGET_SIZE = config["detection_cfg"]["x_resize"]             # target x size based on model training
Y_TARGET_SIZE = config["detection_cfg"]["y_resize"]             # target y size based on model training
CLASS_FILTER  = config["detection_cfg"]["class_filter"]         # class filter on classification output
CLASSIFIC_TH  = config["detection_cfg"]["classif_th"]           # treshold on classification score
FILTER_OBJ_NR = config["detection_cfg"]["filter_on_detect_nr"]  # filter on detected object number with higher score

classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]

timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_PATH = f"{OUTPUT_VIDEO_PATH}_{timestamp_str}.mp4"

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
    

    #################### FOR OUTPUT VIDEO SAVING PURPOSE ONLY #######################
    if FLAG_OUTPUT_SAVE:
        # === VIDEO WRITER INITIALIZATION ===
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    #################################################################################


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

        #################### FOR OUTPUT VIDEO SAVING PURPOSE ONLY #######################
        if FLAG_OUTPUT_SAVE:
            # === DRAWING RED 'X' IF TARGET VALID ===
            if (x_target != -1) and (y_target != -1):
                frame_gbr = draw_red_cross(frame_gbr, (x_target, y_target))
        #################################################################################

        if (x_target == -1 & y_target == -1):
            # keep the actual robot state and do not move
            robot_updated_state = robot_actual_state

        else:
            # update robot state after movement
            robot_updated_state = robot_control(robot_actual_state, x_target, y_target)

        robot_actual_state = robot_updated_state
        frame_count += 1

        #################### FOR OUTPUT VIDEO SAVING PURPOSE ONLY #######################
        if FLAG_OUTPUT_SAVE:
            # === WRITE THE FRAME ===
            out.write(frame_gbr)
        #################################################################################

    cap.release()
    print("Video processing complete.")

    #################### FOR OUTPUT VIDEO SAVING PURPOSE ONLY #######################
    if FLAG_OUTPUT_SAVE:
        out.release()
    #################################################################################

if __name__ == "__main__":
    main()







