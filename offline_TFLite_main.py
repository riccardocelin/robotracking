# first attempt to main structure for classification management - TFLite model
#
# workflow:
# - setup video path (later on there will be a camera support)
# - cycle on video frame (with a customizable skippable frames nr)
# - for each not skipped frame, run computer vision model
# - get the coordinates wrt the center of the camera
# - call to empty functions (only signatures by far) for robot control and servo movement
#
# activate virtual env: source venv/bin/activate (in VS terminal)
#
# # TODO:
# - EDGE_DEV_FLAG management in config (online version)
# - TFlite works with numpy arrays, not tensors: check if the cv_grb_frame -> tf_rgb_frame prepro is necessary or not

# ====================================
# === IMPORT PY PKGS AND FUNCTIONS ===
import cv2
#import time
#import numpy as np
import json
import datetime
from pathlib import Path
import time

from utilities.computer_vision_utils import *
from utilities.robot_control_utils import *

EDGE_DEV_FLAG = True
#EDGE_DEV_FLAG = True
if (EDGE_DEV_FLAG):
    from tflite_runtime.interpreter import Interpreter
    # need to install this in raspberry environment with the following line (on python3.9):
    # pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.10.0-cp39-cp39-linux_aarch64.whl
else:
    from tensorflow.lite.python.interpreter import Interpreter

# ====================================
# ====================================



# ===================================
# =========SUPPORT FUNCTIONS ========

# === TFLITE IMPORT FUNCTION ===
def load_interpreter(model_path, num_threads=1):
    num_threads = max(4, num_threads)
    interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

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

# EDGE_DEV_FLAG = config["deployment_cfg"]["enable_edge_device"]  # flag for enabling usage on raspberry #TODO

classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]

timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_PATH = f"{OUTPUT_VIDEO_PATH}_{timestamp_str}.mp4"

# =============================
# =============================


# === INTERPRETER LOADING ===
current_folder = Path(__file__).parent.resolve() # Get the directory of the current script
model_full_path = current_folder / MODEL_PATH
tflite_interpreter = load_interpreter(model_full_path, num_threads=4) # tflite interpreter

# =============================
# =============================



# === MAIN LOOP ===
def main():

    input_video_path = str(current_folder/VIDEO_PATH)

    # set robot init state
    robot_actual_state = robot_state_init()
    robot_updated_state = robot_actual_state

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return
    
    if FLAG_OUTPUT_SAVE:
        # === VIDEO WRITER INITIALIZATION ===
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(current_folder/OUTPUT_VIDEO_PATH), fourcc, fps, (frame_width, frame_height))
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

            # Resize the frame to the model's input size (320x320)
            frame_resized = cv2.resize(frame_rgb, (X_TARGET_SIZE, Y_TARGET_SIZE))

            # Add batch dimension and ensure dtype uint8
            frame_resized_uint8_batched = np.expand_dims(frame_resized, axis=0).astype(np.uint8)

            # get raw classification from mobilenet model
            # INFO FROM NETRON website:
            # INPUT: tensor uint8[1,320,320,3]
            start_t = time.time()
            boxes, scores, classes, num_detections =  object_detection_tflite_fcn(tflite_interpreter, frame_resized_uint8_batched)
            end_t = time.time()
            print(end_t - start_t)

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







