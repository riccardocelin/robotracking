import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from picamera2 import Picamera2
from types import SimpleNamespace
import builtins

class RoboTracker:
    def __init__(self, CV_MODEL_PRJ_PATH = "",
                 X_TARGET_SIZE = 320, Y_TARGET_SIZE = 320,
                 CLASS_FILTER = 37, CLASSIFIC_TH = 0.2, FILTER_OBJ_NR = 1):

        """
        Tracker contructor function

        # -- default params for sport ball deteciton on mobile net sdd cv model
        X_TARGET_SIZE = 320     # target x size based on model training - mobile net training x size
        Y_TARGET_SIZE = 320     # target y size based on model training - mobile net training y size
        CLASS_FILTER  = 37      # class filter on classification output
        CLASSIFIC_TH  = 0.2     # treshold on classification score
        FILTER_OBJ_NR = 1       # filter on detected object number with higher score
        """

        self.track_state    = SimpleNamespace() # quick and dirty, need to define another class for control stuff
        self.detection      = SimpleNamespace() # quick and dirty, need to define another class for detection stuff

        # Tracker actual state
        self.track_state.prev_target_coord = (-1, -1)
        self.track_state.servo1_deg = 0
        self.track_state.servo2_deg = 0
        self.track_state.is_servo_moving = False
        
        # detection settings
        self.detection.__x_target_size = X_TARGET_SIZE
        self.detection.__y_target_size = Y_TARGET_SIZE
        self.detection.__model = self.__get_cv_model(CV_MODEL_PRJ_PATH)
        self.detection.__camera = self.__init_Pi_camera()
        self.detection.x_actual_raw_target = -1
        self.detection.y_actual_raw_target = -1
        self.detection.x_actual_filt_target = -1
        self.detection.y_actual_filt_target = -1
        self.detection.classification_filter_param = [CLASS_FILTER, CLASSIFIC_TH, FILTER_OBJ_NR]
        self.detection.flutt_filt_thresh = 5
        self.detection.LP_filt_tresh = 0.5
        self.detection.frame_orig_h = -1
        self.detection.frame_orig_w = -1
        self.detection.actual_raw_results = [-1,-1,-1,-1]
        self.detection.actual_filt_results = [-1,-1,-1,-1]

    #########################
    # Computer vision methods

    # === LOAD MODEL ===
    def __get_cv_model(self, model_path_in_prj):
        print("-- computer vision model loading")
        current_folder = Path(__file__).parent.resolve()
        model_full_path = current_folder / model_path_in_prj
        return tf.saved_model.load(str(model_full_path))

    # === INITIALIZE CAMERA ===
    def __init_Pi_camera(self, FPS = 30):

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
                        main={"size": (self.detection.__x_target_size, self.detection.__y_target_size)},
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
        self.detection.frame_orig_h, self.detection.frame_orig_w, _ = frame.shape

        return frame

    # === FRAME PREPROCESSING ===
    def frame_cv_to_tf_colours(frame_bgr):
        # Convert from BGR (OpenCV) to RGB (TensorFlow)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb


    # === FRAME PREPROCESSING ===
    def prepro_frame(frame, x_size = 320, y_size = 320):

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

        self.detection.actual_raw_results = [boxes, scores, classes, num_detections]

        self.filter_on_detection_nr()

        self.get_object_center_for_tracking()


    def get_actual_target_coords(self, GET_RAW = True):
        if GET_RAW:
            return (self.detection.x_actual_raw_target, self.detection.y_actual_raw_target)
        else:
            return (self.detection.x_actual_filt_target, self.detection.y_actual_filt_target)
        
        
    def set_actual_target_coords(self, x=-1, y=-1, SET_RAW = True):
        if SET_RAW:
            self.detection.x_actual_raw_target = x, self.detection.y_actual_raw_target = y
        else:
            self.detection.x_actual_filt_target = x, self.detection.y_actual_filt_target = y


    def reset_actual_target_coord(self, x=-1, y=-1):
        self.set_actual_target_coords(SET_RAW = True)
        self.set_actual_target_coords(SET_RAW = False)

    

    def get_actual_detection_results(self, GET_RAW = True):
        if GET_RAW:
            # detection.actual_raw_results = [boxes, scores, classes, num_detections]
            return self.detection.actual_raw_results
        else:
            # detection.actual_filt_results = [boxes_filt, scores_filt, classes_filt, num_detections_filt]
            return self.detection.actual_filt_results
    

    def get_classification_params(self):
        # classif_results = [boxes, scores, classes, num_detections]
        return self.detection.classification_filter_param


    def is_target_detected_on_frame(self):
        x,y = self.get_actual_target_coords(GET_RAW = True)
        return (x != -1 and y != -1)
        

    # === CLASSIFICATION RESULTS FILTERING ===
    def filter_on_detection_nr(self):

        boxes, scores, classes, num_detections = self.get_actual_detection_results(GET_RAW = True)

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

        self.detection.actual_filt_results = [boxes_filt, scores_filt, classes_filt, num_detections_filt]


    # === GET OBJECT CENTER ===
    def get_object_center_for_tracking(self):

        h_orig_img = self.detection.frame_orig_h
        w_orig_img = self.detection.frame_orig_w

        boxes_filt, _, _, _ = self.get_actual_detection_results(GET_RAW=False)

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

            # filtering coord
            self.process_coordinates()

        self.set_actual_target_coords(object_center[0], object_center[1], SET_RAW=True)



    # === HELPER FCN IN CASE OF OUTPUT WRITING ===
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


    def process_coordinates(self, act_coord):
        """
        Processes the detected coordinates by applying a low-pass filter
        and a flutter limiter.
        """

        # Apply flutter limiter to ignore small changes below threshold
        filt1_coord = self._detection_flutt_limiter(
            filt1_coord,
            self.target_coord,
            self.detection_filter_thresh
        )

        # Determine if servo needs to move (only if coordinates have changed)
        servo_has_to_move = filt1_coord != self.target_coord

        # Update the target coordinates
        self.target_coord = filt1_coord

        # Apply low-pass filter to smooth coordinate fluctuations
        filt2_coord = self.detection_filter(act_coord)

        return filt2_coord, servo_has_to_move


    def __detection_flutt_limiter(self):
        # Limits small fluctuations (flutter) in coordinates.
        # Keeps the previous value if the change is below a given threshold.

        x_act, y_act = self.get_actual_target_coords(GET_RAW=True)
        x_prev, y_prev = get_prev_target_coords(GET_RAW=True)

        x_diff = abs(x_act - x_prev)
        y_diff = abs(y_act - y_prev)

        x_new = x_act if x_diff > self.detection.flutt_filt_thresh else x_prev
        y_new = y_act if y_diff > self.detection.flutt_filt_thresh else y_prev

        if builtins.DEBUG: print("### flutter filter has been applied")

        self.x_actual_raw  =

    
    """
    def get_prev_target_coord(self):
        return (self.detection.x_last_coord, self.detection.y_last_coord)


    def detection_filter(self):
        #Applies a low-pass filter to the detected coordinates to reduce noise.
        alpha = 0.5

        x_prev, y_prev = self.target_coord
        x_act, y_act = self.last_coord

        x_new = alpha * x_act + (1 - alpha) * x_prev
        y_new = alpha * y_act + (1 - alpha) * y_prev

        return (x_new, y_new)




    
    #########################

    #############################
    # Tracker and control methods

    # === ROBOT CONTROL FUNCTION ===
    def robot_control(self, actual_state, x,y):
        # Implement servo control logic here
        print("Executed robot_control with: (%d,%d)" %(x,y))
        self.track_state.updated_state = self.track_state.actual_state
        return self.track_state.updated_state

    # === ROBOT CONTROL FUNCTION ===
    def robot_state_init(self):
        # Implement robot state initialization logic here
        print("Executed robot state init")
        self.track_state.state_t0 = []
        return self.track_state.state_t0


    #############################
        
    """
