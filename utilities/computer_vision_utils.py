import cv2
import tensorflow as tf
import numpy as np

# === LOAD MODEL ===
def load_tf_saved_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


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
def object_detection_fcn(model, tf_frame):

     # Run inference
    model_infer_fcn = model.signatures['serving_default']
    output_dict = model_infer_fcn(tf_frame)

    # Number of detections
    num_detections = int(output_dict['num_detections'][0])

    # Get detection boxes, scores, and classes
    boxes = output_dict['detection_boxes'][0][:num_detections].numpy()  # (ymin, xmin, ymax, xmax)
    scores = output_dict['detection_scores'][0][:num_detections].numpy()
    classes = output_dict['detection_classes'][0][:num_detections].numpy().astype(int)

    return boxes, scores, classes, num_detections


# === TFLITE OUTPUT DEQUANTIZATION ===
def dequantize(tensor, scale, zero_point):
    return (tensor.astype(np.float32) - zero_point) * scale


# === FRAME CLASSIFICATION TFLITE INTERPRETER ===
def object_detection_tflite_fcn(interpreter, input_data):
    # Get input & output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Convert Tensor to NumPy (TFLite needs NumPy input)
    #input_data = tf_frame.numpy().astype(input_details[0]['dtype']) # uint8

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensors
    boxes_output = output_details[4]
    scale, zero_point = boxes_output['quantization']
    boxes = dequantize(interpreter.get_tensor(boxes_output['index'])[0], scale, zero_point)

    classes_output = output_details[5]
    scale, zero_point = classes_output['quantization']
    classes = dequantize(interpreter.get_tensor(classes_output['index'])[0], scale, zero_point).astype(int)

    scores_output = output_details[6]
    scale, zero_point = scores_output['quantization']
    scores = dequantize(interpreter.get_tensor(scores_output['index'])[0], scale, zero_point)

    num_detections = boxes.shape[0]

    return boxes, scores, classes, num_detections


# === CLASSIFICATION RESULTS FILTERING ===
def filter_on_detection_nr(classif_results, classif_params):

    # classif_results = [boxes, scores, classes, num_detections]
    boxes, scores, classes, num_detections = classif_results

    # check for class_filter
    # check for scores
    # order and check for keeping only the first 'filter_on_detect_nr'
    class_filter, classif_th, filter_on_detect_nr = classif_params

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

    return boxes_filt, scores_filt, classes_filt, num_detections_filt



# === GET OBJECT CENTER ===
def get_object_center_for_tracking(boxes, h_orig_img, w_orig_img):

    if not boxes:
        object_center = [-1, -1]
    else:
        box = boxes[0]
        startY = int(box[0]*h_orig_img)
        startX = int(box[1]*w_orig_img)
        endY   = int(box[2]*h_orig_img)
        endX   = int(box[3]*w_orig_img)

        Y_center = int((endY-startY)/2) + startY
        X_center = int((endX-startX)/2) + startX

        object_center = [X_center, Y_center]

    return object_center