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
SIMULATION = True

# ====================================
# === IMPORT PY PKGS AND FUNCTIONS ===
import cv2
import time

from utilities.Detection import Detection

if SIMULATION:
    from Test.Camera_simulator import CameraSimulator
    from Test.Control_simulator import ControlSimulator
else:
    from utilities.Control import Control
    from utilities.Camera import Camera



# =============================
# === SETTINGS DEFINITION ===
MODEL_PATH  = "computer_vision/ssd_mobilenet_v2_320x320_coco17_tpu-8/TFLite/prepro_model_nodynamicinput/saved_model" # Path to the saved TensorFlow model
filter_flag = True
Control_algorithm = "P"   # control algorithm to be used ("P", "PI", "PID", etc.)
focal_length = 800  # camera focal length in pixels
# =============================
# =============================


# === MAIN LOOP ===
def main():

    print(MODEL_PATH)
    detector = Detection(MODEL_PATH, FILTER_FLAG = filter_flag)
    if SIMULATION:
        camera = CameraSimulator()
        control_obj = ControlSimulator(Control_type = Control_algorithm, focal_length = focal_length)
    else:
        camera = Camera()
        control_obj = Control(detector, Control_type = Control_algorithm, focal_length = camera.focal_length)

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
            print(f"\n### Inference time for actual frame: {end_t - start_t:.3f} sec")
            x, y = detector.get_actual_target_coords(GET_RAW=True)
            print(f"### Raw target coordinates x,y: {x}, {y}")
            x, y = detector.get_actual_target_coords(GET_RAW=False)
            print(f"### Filtered target coordinates x,y: {x}, {y}")

            # raw object target found (draw raw center obj and filtered target coords)
            if detector.is_target_detected_on_frame():
                frame = detector.draw_cross_on_frame(frame,
                [detector.get_actual_target_coords(GET_RAW=True), 
                detector.get_actual_target_coords(GET_RAW=False)], 
                [(255,0,0), (0,255,0)])
        
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
            target_coords = detector.get_actual_target_coords(GET_RAW=False)
            control_obj.update_target_coords(target_coords[0], target_coords[1])
            u_z, u_x = control_obj.robot_control()
            
            # ACTUATION OF CONTROL ACTION
            if DEBUG:
                print(f"### Control action: u_z,u_x: {u_z}, {u_x}")

            if SIMULATION:
                camera.set_camera_rotation(rot_z=u_z, rot_x=u_x)
            # else:
                # ACTUATION TODO
            # # tracker.robot_updated_state = tracker.robot_control(tracker.robot_actual_state, x_target_f2, y_target_f2)
            
        detector.reset_actual_target_coord() # clean actual state for the next frame (does not reset pstep target coords)


    # Clean up
    detector.__camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()









