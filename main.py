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

DEBUG  = True   # flag for output video display enable (online debug purpose)

# ====================================
# === IMPORT PY PKGS AND FUNCTIONS ===
import cv2
import time

from utilities.Detection import Detection

if DEBUG:
    from Test.Camera_simulator import CameraSimulator
else:
    from utilities.Control import Control
    from utilities.Camera import Camera



# =============================
# === SETTINGS DEFINITION ===
MODEL_PATH  = "computer_vision/ssd_mobilenet_v2_320x320_coco17_tpu-8/TFLite/prepro_model_nodynamicinput/saved_model" # Path to the saved TensorFlow model
filter_flag = True
Control_algorithm = "P"   # control algorithm to be used ("P", "PI", "PID", etc.)
# =============================
# =============================


# === MAIN LOOP ===
def main():

    print(MODEL_PATH)
    detector = Detection(MODEL_PATH, FILTER_FLAG = filter_flag)
    if DEBUG:
        camera = CameraSimulator()
    else:
        camera = Camera()

    print("Starting video loop...")

    while True:
        frame = camera.get_camera_frame() # get frame from camera module

        # get raw classification from cv model
        start_t = time.time()

        # === OBJECT DETECTION ON FRAME ===
        detector.object_detection_fcn(frame)

        end_t = time.time()

        ########################## FOR DEBUG PURPOSE ONLY ###############################
        if DEBUG:
            print(f"### Inference time for actual frame: {end_t - start_t:.3f} sec")
            x, y = detector.get_actual_target_coords(GET_RAW=True)
            print(f"### Raw target coordinates x,y: {x}, {y}")
            x, y = detector.get_actual_target_coords(GET_RAW=False)
            print(f"### Filtered target coordinates x,y: {x}, {y}")

            # raw object target found (draw raw center obj and filtered target coords)
            if detector.is_target_detected_on_frame():
                frame = detector.draw_cross_on_frame(frame, [detector.get_actual_target_coords(GET_RAW=True), detector.get_actual_target_coords(GET_RAW=False)], [(255,0,0), (0,255,0)])
        
            cv2.imshow("Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # cv2 requires bgr frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #################################################################################

        if (detector.is_target_detected_on_frame() == False):
            # keep the actual robot state (and do not move? or continue moving if will work on separated threads)
            # tracker.robot_updated_state = tracker.robot_actual_state

            # TODO
            pass

        else:
            # control_obj = Control(detector, Control_type = Control_algorithm)
            pass

            # ACTUATION OF CONTROL ACTION
            # # tracker.robot_updated_state = tracker.robot_control(tracker.robot_actual_state, x_target_f2, y_target_f2)
            

        detector.reset_actual_target_coord() # clean actual state for the next frame (does not reset pstep target coords)

    # Clean up
    detector.__camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()









