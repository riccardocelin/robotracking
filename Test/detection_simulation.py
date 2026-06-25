import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import time
import numpy as np

from src.Detection import Detection
from src.simulators.CameraSimulator import CameraSimulator


# === MAIN LOOP ===
def main():

    detector = Detection(FILTER_FLAG = True)
    camera = CameraSimulator()

    print("Starting video loop...")

    frame_count = 0
    total_time = 0
    avg_time = 0

    while True:
        frame = camera.get_camera_frame() # get frame from camera module

        # get raw classification from cv model
        start_t = time.time()

        # === OBJECT DETECTION ON FRAME ===
        detector.object_detection_fcn(frame)

        end_t = time.time()

        inference_time = end_t - start_t

        # update running average
        frame_count += 1
        total_time += inference_time
        avg_time = total_time / frame_count

        print(f"\n### Inference time for actual frame: {inference_time:.3f} sec")
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
            

        cv2.putText(
            img = frame,
            text = f"Avg detection time per frame: {avg_time:.4f} s",
            org = (10, 30),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType = cv2.LINE_AA
        )
    
        cv2.imshow("Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # cv2 requires bgr frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        camera.set_camera_rotation(
            rot_z=np.deg2rad(5),
            rot_x=np.deg2rad(5)
        )
            
        detector.reset_actual_target_coord() # clean actual state for the next frame (does not reset pstep target coords)


    # Clean up
    detector.__camera.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()









