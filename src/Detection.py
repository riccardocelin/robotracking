import cv2
import numpy as np
import requests
import time


class Detection:
    def __init__(self,
                 X_TARGET_SIZE = 320, Y_TARGET_SIZE = 320,
                 CLASS_FILTER = 37, CLASSIFIC_TH = 0.1, FILTER_OBJ_NR = 1, FILTER_FLAG = False):

        """
        Tracker contructor function

        # -- default params for sport ball deteciton on mobile net sdd cv model
        X_TARGET_SIZE = 320     # target x size based on model training - mobile net training x size
        Y_TARGET_SIZE = 320     # target y size based on model training - mobile net training y size
        CLASS_FILTER  = 37      # class filter on classification output
        CLASSIFIC_TH  = 0.1     # treshold on classification score
        FILTER_OBJ_NR = 1       # filter on detected object number with higher score
        """

        DEF_VAL = -1

        # detection settings
        self.__x_target_size = X_TARGET_SIZE
        self.__y_target_size = Y_TARGET_SIZE
        self.x_actual_raw_target = DEF_VAL
        self.y_actual_raw_target = DEF_VAL
        self.x_actual_filt_target = DEF_VAL
        self.y_actual_filt_target = DEF_VAL
        self.x_pstep_filt_target = DEF_VAL
        self.y_pstep_filt_target = DEF_VAL
        self.class_filter = CLASS_FILTER
        self.classific_th = CLASSIFIC_TH
        self.filter_on_detect_nr = FILTER_OBJ_NR
        self.flutt_filt_thresh = 5
        self.LP_filt_tresh = 0.5
        self.frame_orig_h = DEF_VAL
        self.frame_orig_w = DEF_VAL
        self.actual_raw_results = [DEF_VAL,DEF_VAL,DEF_VAL,DEF_VAL]
        self.actual_filt_results = [DEF_VAL,DEF_VAL,DEF_VAL,DEF_VAL]
        self.filter_flag = FILTER_FLAG

    #########################
    # Computer vision methods

    def store_camera_frame_info(self, frame):

        # store initial frame shape
        self.frame_orig_h, self.frame_orig_w, _ = frame.shape

        return frame     


    # === FRAME CLASSIFICATION ===
    def object_detection_fcn(self, frame):

        # get frame from camera module
        if self.frame_orig_h == -1 and self.frame_orig_w == -1:
            self.store_camera_frame_info(frame)

        # target size based on model training
        x_target_size = self.__x_target_size
        y_target_size = self.__y_target_size

        if frame.shape[0] != x_target_size and frame.shape[1] != y_target_size:
            frame = cv2.resize(frame, (x_target_size, y_target_size)) # resize frame to model input size

        h_input_frame, w_input_frame, c_input_frame = frame.shape

        params = {
            "h_origin": self.frame_orig_h,
            "w_origin": self.frame_orig_w,
            "h_input_frame": h_input_frame,
            "w_input_frame": w_input_frame,
            "c_input_frame": c_input_frame,
            "class_filter": self.class_filter,
            "classific_th": self.classific_th,
            "filter_on_detect_nr": self.filter_on_detect_nr
        }

        start_t1 = time.time()
        response = requests.post("http://127.0.0.1:8000/detect", data = frame.tobytes(), params = params)
        end_t1 = time.time()
        time_inf = end_t1-start_t1
        print(f"time inference inside Detection.py: {time_inf}")

        results = response.json()

        self.set_actual_target_coords(results["object_center"][0], results["object_center"][1], SET_RAW=True)

        self.postpro_coordinates() # coords filtering and pstep update


    def get_actual_target_coords(self, GET_RAW = True):
        if GET_RAW:
            return (self.x_actual_raw_target, self.y_actual_raw_target)
        else:
            return (self.x_actual_filt_target, self.y_actual_filt_target)
        
        
    def set_actual_target_coords(self, x=-1, y=-1, SET_RAW = True):
        if SET_RAW:
            self.x_actual_raw_target = x
            self.y_actual_raw_target = y
        else:
            self.x_actual_filt_target = x
            self.y_actual_filt_target = y
            

    def get_pstep_target_coords(self):
        return (self.x_pstep_filt_target, self.y_pstep_filt_target)
    
    def set_pstep_target_coords(self, x=-1, y=-1):
        self.x_pstep_filt_target = x
        self.y_pstep_filt_target = y

    def reset_actual_target_coord(self, x=-1, y=-1):
        self.set_actual_target_coords(SET_RAW = True)
        self.set_actual_target_coords(SET_RAW = False)

    
    def get_classification_params(self):
        # classif_results = [boxes, scores, classes, num_detections]
        return self.classification_filter_param


    def is_target_detected_on_frame(self):
        x,y = self.get_actual_target_coords(GET_RAW = True)
        return (x != -1 and y != -1)
        

    def draw_cross_on_frame(self, frame, coords_array = [(-1,-1)], color_array = [(255, 0, 0)], size=15, thickness=3):
        """
        Draws a coloured 'X' centered at 'position' on the frame.
        """

        if (self.is_target_detected_on_frame() == False):
            return frame
        
        if (len(coords_array) != len(color_array)):
            return frame

        #color = (255, 0, 0)  # Red color in BGR

        # Check and fix common OpenCV incompatibilities
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame is not a NumPy array.")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        height, width = frame.shape[:2]

        # Clamp coordinates to avoid drawing outside image bounds

        for i in range(len(coords_array)):

            color = color_array[i]
            x, y = coords_array[i]

            x_min = max(x - size, 0)
            y_min = max(y - size, 0)
            x_max = min(x + size, width - 1)
            y_max = min(y + size, height - 1)

            cv2.line(frame, (x_min, y_min), (x_max, y_max), color, thickness)
            cv2.line(frame, (x_min, y_max), (x_max, y_min), color, thickness)

        return frame


    def postpro_coordinates(self):
        """
        Processes the detected coordinates by applying a low-pass filter
        and a flutter limiter.
        """
        if not self.is_target_detected_on_frame():
            return # no raw coords to be processed

        if self.filter_flag:
            # Apply flutter limiter to ignore small changes below threshold
            # self.__detection_flutt_limiter() # update raw detection and filter actual detection?

            # Apply low pass filter on filt coords and update filt detection
            self.__detection_filter()


    def __detection_flutt_limiter(self):
        # Limits small fluctuations (flutter) in coordinates.
        # Keeps the previous value if the change is below a given threshold.

        if not self.is_target_detected_on_frame(): return # no raw coords to be processed

        x_act, y_act = self.get_actual_target_coords(GET_RAW=True)
        x_prev = self.x_pstep_filt_target
        y_prev = self.y_pstep_filt_target

        x_diff = abs(x_act - x_prev)
        y_diff = abs(y_act - y_prev)

        x_new = x_act if x_diff > self.flutt_filt_thresh else x_prev
        y_new = y_act if y_diff > self.flutt_filt_thresh else y_prev

        self.set_pstep_target_coords(x_act, y_act)  # update pstep raw target value
        self.set_actual_target_coords(x_new, y_new, SET_RAW=True) # update actual raw target value


    def __detection_filter(self, w_actual = 0.7, w_prev = 0.3):
        #Applies a low-pass filter to the detected coordinates to reduce noise.

        if not self.is_target_detected_on_frame(): return # no coords to be processed

        if self.get_pstep_target_coords() == (-1,-1):
            self.set_pstep_target_coords(self.x_actual_raw_target, self.y_actual_raw_target) # set filt coords as raw target if it was at default

        x_act = self.x_actual_raw_target
        y_act = self.y_actual_raw_target
        x_prev = self.x_pstep_filt_target
        y_prev = self.y_pstep_filt_target

        x_new = (w_actual * x_act + w_prev * x_prev)/(w_actual + w_prev)
        y_new = (w_actual * y_act + w_prev * y_prev)/(w_actual + w_prev)

        self.set_actual_target_coords(int(x_new), int(y_new), SET_RAW=False)  # update actual filt target value
        self.set_pstep_target_coords(int(x_new), int(y_new)) # update pstep filt target value
