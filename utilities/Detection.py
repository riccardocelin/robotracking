import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from picamera2 import Picamera2
import builtins

class Detection:
    def __init__(self, CV_MODEL_PRJ_PATH = "",
                 X_TARGET_SIZE = 320, Y_TARGET_SIZE = 320,
                 CLASS_FILTER = 37, CLASSIFIC_TH = 0.2, FILTER_OBJ_NR = 1, FILTER_FLAG = False):

        """
        Tracker contructor function

        # -- default params for sport ball deteciton on mobile net sdd cv model
        X_TARGET_SIZE = 320     # target x size based on model training - mobile net training x size
        Y_TARGET_SIZE = 320     # target y size based on model training - mobile net training y size
        CLASS_FILTER  = 37      # class filter on classification output
        CLASSIFIC_TH  = 0.2     # treshold on classification score
        FILTER_OBJ_NR = 1       # filter on detected object number with higher score
        """

        # detection settings
        self.__x_target_size = X_TARGET_SIZE
        self.__y_target_size = Y_TARGET_SIZE
        self.__model = self.__get_cv_model(CV_MODEL_PRJ_PATH)
        self.__camera = self.__init_Pi_camera()
        self.x_actual_raw_target = -1
        self.y_actual_raw_target = -1
        self.x_actual_filt_target = -1
        self.y_actual_filt_target = -1
        self.x_pstep_raw_target = -1
        self.y_pstep_raw_target = -1
        self.x_pstep_filt_target = -1
        self.y_pstep_filt_target = -1
        self.classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]
        self.flutt_filt_thresh = 5
        self.LP_filt_tresh = 0.5
        self.frame_orig_h = -1
        self.frame_orig_w = -1
        self.actual_raw_results = [-1,-1,-1,-1]
        self.actual_filt_results = [-1,-1,-1,-1]
        self.filter_flag = FILTER_FLAG

    #########################
    # Computer vision methods

    # === LOAD MODEL ===
    def __get_cv_model(self, model_path_in_prj):
        print("-- computer vision model loading")
        current_folder = Path(__file__).parent.resolve()
        model_full_path = current_folder.parent / model_path_in_prj

        return tf.saved_model.load(str(model_full_path))

    # === INITIALIZE CAMERA ===
    def __init_Pi_camera(self, FPS = 30):

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
                        main={"size": (self.__x_target_size, self.__y_target_size)},
                        controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))}
                    )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        return picam2

    def get_camera_frame(self):
        frame_rgb = self.__camera.capture_array()
        
        frame = frame_rgb
        if frame.shape[-1] == 4:
            frame = frame[..., :3]  # remove eventual transparency from picam2 module

        # store initial frame shape
        self.frame_orig_h, self.frame_orig_w, _ = frame.shape

        return frame

    # === FRAME PREPROCESSING ===
    def frame_cv_to_tf_colours(self, frame_bgr):
        # Convert from BGR (OpenCV) to RGB (TensorFlow)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb


    # === FRAME PREPROCESSING ===
    def prepro_frame(self, frame, x_size = 320, y_size = 320):

        # Decode image (this keeps dtype uint8)
        tf_frame_uint8 = tf.convert_to_tensor(frame, dtype=tf.uint8)

        # Resize to x_size * y_size — what SSD MobileNet was trained on
        tf_frame_resized_float32 = tf.image.resize(tf_frame_uint8, (x_size, y_size))

        # Cast to uint8 (since resize converts to float32)
        tf_frame_resized_uint8 = tf.cast(tf_frame_resized_float32, dtype=tf.uint8)

        # Add batch dimension: (1,x_size, y_size, 3)
        tf_frame_resized_uint8_batched = tf.expand_dims(tf_frame_resized_uint8, axis=0) # Add batch dimension: (1, x_dim, y_dim, 3)

        return tf_frame_resized_uint8_batched, tf_frame_resized_float32


    # === FRAME CLASSIFICATION ===
    def object_detection_fcn(self, frame):

        # frame preprocessing for TF inference
        tf_frame_resized_uint8_batched, tf_frame_resized_float32 = self.prepro_frame(frame)

        # Run inference
        model_infer_fcn = self.__model.signatures['serving_default']
        
        output_dict = model_infer_fcn(tf_frame_resized_uint8_batched)

        # Number of detections
        num_detections = int(output_dict['num_detections'][0])

        # Get detection boxes, scores, and classes
        boxes = output_dict['detection_boxes'][0][:num_detections].numpy()  # (ymin, xmin, ymax, xmax)
        scores = output_dict['detection_scores'][0][:num_detections].numpy()
        classes = output_dict['detection_classes'][0][:num_detections].numpy().astype(int)

        self.actual_raw_results = [boxes, scores, classes, num_detections]

        # filtering on the detection results
        self.filter_on_detection_nr()

        object_center = self.get_object_center_for_tracking()

        self.set_actual_target_coords(object_center[0], object_center[1], SET_RAW=True)

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
            

    def get_pstep_target_coords(self, GET_RAW = True):
        if GET_RAW:
            return (self.x_pstep_raw_target, self.y_pstep_raw_target)
        else:
            return (self.x_pstep_filt_target, self.y_pstep_filt_target)
    
    def set_pstep_target_coords(self, x=-1, y=-1, SET_RAW = True):
        if SET_RAW:
            self.x_pstep_raw_target = x
            self.y_pstep_raw_target = y
        else:
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
        

    # CLASSIFICATION RESULTS FILTERING
    def filter_on_detection_nr(self):

        boxes, scores, classes, num_detections = self.actual_raw_results

        # check for class_filter
        # check for scores
        # order and check for keeping only the first 'filter_on_detect_nr'
        class_filter, classif_th, filter_on_detect_nr = self.get_classification_params()

        boxes_filt = []
        scores_filt = []
        classes_filt = []

        for i in range(num_detections):
            if ((class_filter == -1) or (class_filter != -1 and classes[i] == class_filter)):
                if scores[i] >= classif_th:
                    boxes_filt.append(boxes[i])
                    scores_filt.append(scores[i])
                    classes_filt.append(classes[i])

        # Sort by scores descending
        sorted_indices = sorted(range(len(scores_filt)), key=lambda k: scores_filt[k], reverse=True)
        boxes_filt = [boxes_filt[i] for i in sorted_indices]
        scores_filt = [scores_filt[i] for i in sorted_indices]
        classes_filt = [classes_filt[i] for i in sorted_indices]

        # Apply filter_on_detect_nr if needed
        if filter_on_detect_nr != -1:
            boxes_filt = boxes_filt[:filter_on_detect_nr]
            scores_filt = scores_filt[:filter_on_detect_nr]
            classes_filt = classes_filt[:filter_on_detect_nr]

        num_detections_filt = len(boxes_filt)

        self.actual_filt_results = [boxes_filt, scores_filt, classes_filt, num_detections_filt]


    # GET OBJECT CENTER FOR TRACKING
    def get_object_center_for_tracking(self):

        h_orig_img = self.frame_orig_h
        w_orig_img = self.frame_orig_w

        boxes_filt, _, _, _ = self.actual_filt_results

        if not boxes_filt:
            object_center = [-1, -1]
        else:
            box = boxes_filt[0]
            startY = int(box[0]*h_orig_img)
            startX = int(box[1]*w_orig_img)
            endY   = int(box[2]*h_orig_img)
            endX   = int(box[3]*w_orig_img)

            Y_center = int((endY-startY)/2) + startY
            X_center = int((endX-startX)/2) + startX

            object_center = [X_center, Y_center]

        return object_center
    

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
            self.__detection_flutt_limiter() # update raw detection and filter actual detection?

            # Apply low pass filter on filt coords and update filt detection
            self.__detection_filter()


    def __detection_flutt_limiter(self):
        # Limits small fluctuations (flutter) in coordinates.
        # Keeps the previous value if the change is below a given threshold.

        if not self.is_target_detected_on_frame(): return # no raw coords to be processed

        x_act, y_act = self.get_actual_target_coords(GET_RAW=True)
        x_prev, y_prev = self.get_pstep_target_coords(GET_RAW=True)

        x_diff = abs(x_act - x_prev)
        y_diff = abs(y_act - y_prev)

        x_new = x_act if x_diff > self.flutt_filt_thresh else x_prev
        y_new = y_act if y_diff > self.flutt_filt_thresh else y_prev

        self.set_pstep_target_coords(x_act, y_act, SET_RAW=True)  # update pstep raw target value
        self.set_actual_target_coords(x_new, y_new, SET_RAW=True) # update actual raw target value


    def __detection_filter(self, alpha = 0.5):
        #Applies a low-pass filter to the detected coordinates to reduce noise.

        if not self.is_target_detected_on_frame(): return # no coords to be processed

        (x_act, y_act) = self.get_pstep_target_coords(GET_RAW=False)
        if (x_act, y_act) == (-1,-1):
            self.set_pstep_target_coords(self.get_actual_target_coords(GET_RAW=True), SET_RAW=False) # set filt coords as raw target if it was at default

        x_prev, y_prev = self.get_actual_target_coords(GET_RAW=True)
        x_act, y_act = self.get_pstep_target_coords(GET_RAW=False)

        x_new = alpha * x_act + (1 - alpha) * x_prev
        y_new = alpha * y_act + (1 - alpha) * y_prev

        return (x_new, y_new)