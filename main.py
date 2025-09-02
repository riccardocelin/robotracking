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
import builtins

from utilities.robotracker import RoboTracker


# =============================
# === GLOBAL VAR DEFINITION ===

MODEL_PATH  = "computer_vision/ssd_mobilenet_v2_320x320_coco17_tpu-8/TFLite/prepro_model_nodynamicinput/saved_model" # Path to the saved TensorFlow model
builtins.DEBUG  = True   # flag for output video display enable (online debug purpose)

# =============================
# =============================



# === MAIN LOOP ===
def main():

    tracker = RoboTracker(MODEL_PATH)

    print("Starting video loop...")

    while True:
        
        frame = tracker.get_camera_frame() # get frame from camera module

        # get raw classification from cv model
        start_t = time.time()

        tracker.object_detection_fcn(tracker.model, frame)

        end_t = time.time()

        ########################## FOR DEBUG PURPOSE ONLY ###############################
        if builtins.DEBUG:
            print(f"### Inference time for actual frame: {end_t - start_t:.3f} sec")
            x, y = tracker.get_actual_target_coords(GET_RAW=True)
            print(f"### Raw target coordinates x,y: {x}, {y}")
            x, y = tracker.get_actual_target_coords(GET_RAW=False)
            print(f"### Filtered target coordinates x,y: {x}, {y}")

            # raw object target found (draw raw center obj and filtered target coords)
            if tracker.is_target_detected_on_frame():
                frame = tracker.draw_cross_on_frame(frame, [tracker.get_actual_target_coords(GET_RAW=True), tracker.get_actual_target_coords(GET_RAW=False)], [(255,0,0), (0,255,0)])
        
            cv2.imshow("Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # cv2 requires bgr frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #################################################################################

        if (tracker.is_target_detected_on_frame() == False):
            # keep the actual robot state (and do not move? or continue moving if will work on separated threads)
            # tracker.robot_updated_state = tracker.robot_actual_state

            # TO DO
            pass

        else:
            # update robot state after movement
            # check if the prev coordinates were valid 
            # # x_target_f1, y_target_f1 = tracker.detection_filter()
            # # x_target_f2, y_target_f2 = tracker.detection_flutt_limiter((), tracker.prev_valid_coord):
            # # tracker.robot_updated_state = tracker.robot_control(tracker.robot_actual_state, x_target_f2, y_target_f2)
            
            # TO DO
            pass

        tracker.reset_actual_target_coord() # clean actual state for the next frame (does not reset pstep target coords)

    # Clean up
    tracker.__camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







