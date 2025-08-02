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
import numpy as np
import json
import datetime

from robotracker import RoboTracker


# =============================
# === GLOBAL VAR DEFINITION ===

MODEL_PATH  = "computer_vision/ssd_mobilenet_v2_320x320_coco17_tpu-8/TFLite/prepro_model_nodynamicinput/saved_model" # Path to the saved TensorFlow model
DEBUG  = True   # flag for output video display enable

# =============================
# =============================



# === MAIN LOOP ===
def main():

    tracker = RoboTracker(MODEL_PATH)

    print("Starting video loop...")

    while True:
        
        frame_rgb = tracker.camera.capture_array()
        
        frame = frame_rgb
        if frame.shape[-1] == 4:
            frame = frame[..., :3]  # remove eventual transparency from picam2 module

        # store initial frame shape
        h, w, _ = frame.shape

        #frame_rgb = frame_cv_to_tf_colours(frame_gbr)
        tf_frame_resized_uint8_batched, tf_frame_resized_float32 = tracker.prepro_frame(frame)

        # get raw classification from cv model
        start_t = time.time()
        boxes, scores, classes, num_detections = tracker.object_detection_fcn(tracker.model, tf_frame_resized_uint8_batched)
        end_t = time.time()
        print(f"Inference time: {end_t - start_t:.3f} sec")

        # filter and order raw classification from mobilenet model based on cfg params
        classification_output = [boxes, scores, classes, num_detections]
        boxes_filt, scores_filt, classes_filt, num_detections_filt = tracker.filter_on_detection_nr(classification_output, tracker.classification_filter_param)
        
        ##### !!!!!!!!!
        # TODO x_target, y_target diventano attributi di classe sotto self.target.x_target ecc e non vengono neanche date in output dalla funzione ma vengono cambiati intenamente gli attributi
        # get single object to be tracked (the first one box is supposed to be the higher score for the class of interest)
        x_target, y_target = tracker.get_object_center_for_tracking(boxes_filt, h, w)
        print(f"Object coordinates x,y: {x_target}, {y_target}")

        ########################## FOR DEBUG PURPOSE ONLY ###############################
        if DEBUG:

            # raw object target found
            if (x_target != -1) and (y_target != -1):
                frame = tracker.draw_cross(frame, (x_target, y_target), (255,0,0))
        
            cv2.imshow("Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # cv2 requires bgr frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #################################################################################

        if (x_target == -1 & y_target == -1):
            # keep the actual robot state and do not move
            tracker.robot_updated_state = tracker.robot_actual_state

        else:
            # update robot state after movement
            # check if the prev coordinates were valid 
            x_target_f1, y_target_f1 = tracker.detection_filter()
            x_target_f2, y_target_f2 = tracker.detection_flutt_limiter((), prev_valid_coord, TRESH_COORD_FILTER):
            tracker.robot_updated_state = tracker.robot_control(tracker.robot_actual_state, x_target_f2, y_target_f2)

    # Clean up
    tracker.camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







