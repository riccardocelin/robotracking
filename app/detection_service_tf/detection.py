import tensorflow as tf
from pathlib import Path

def load_model_signature(model_path_in_prj):

    current_folder = Path(__file__).resolve().parent
    model_full_path = current_folder.parent / model_path_in_prj
    model = tf.saved_model.load(str(model_full_path))

    print(f"Model signatures: {list(model.signatures.keys())}")

    return model.signatures['serving_default']


def object_detection_fcn(model_infer_fcn, frame, params):

    print(f"Input signature: {model_infer_fcn.structured_input_signature}")
    print(f"Output signature: {model_infer_fcn.structured_outputs}")
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")

    frame_tf = tf.convert_to_tensor(frame, dtype=tf.uint8)

    output_dict = model_infer_fcn(frame_tf)

    # Number of detections
    num_detections = int(output_dict['num_detections'][0])

    # Get detection boxes, scores, and classes
    boxes = output_dict['detection_boxes'][0][:num_detections].numpy()  # (ymin, xmin, ymax, xmax)
    scores = output_dict['detection_scores'][0][:num_detections].numpy()
    classes = output_dict['detection_classes'][0][:num_detections].numpy().astype(int)

    # filtering on the detection results
    boxes_filt, scores_filt, classes_filt, num_detections_filt = filter_on_detection_nr(boxes, scores, classes, num_detections, params)

    object_center, detected = get_object_center_for_tracking(boxes_filt, params)

    if detected:
        response = {
            "detected": bool(detected),
            "class": classes_filt[0].astype(int).tolist(),
            "box": boxes_filt[0].astype(int).tolist(),
            "score": scores_filt[0].astype(float).tolist(),
            "object_center": object_center
        }
    else:
        response = {
            "detected": bool(detected),
            "class": -1,
            "box": [-1,-1,-1,-1],
            "score": float(-1),
            "object_center": [-1,-1]
        }

    return response


# CLASSIFICATION RESULTS FILTERING
def filter_on_detection_nr(boxes, scores, classes, num_detections, params):

    boxes_filt = []
    scores_filt = []
    classes_filt = []

    for i in range(num_detections):
        if ((params.class_filter == -1) or (params.class_filter != -1 and classes[i] == params.class_filter)):
            if scores[i] >= params.classif_th:
                boxes_filt.append(boxes[i])
                scores_filt.append(scores[i])
                classes_filt.append(classes[i])

    # Sort by scores descending
    sorted_indices = sorted(range(len(scores_filt)), key=lambda k: scores_filt[k], reverse=True)
    boxes_filt = [boxes_filt[i] for i in sorted_indices]
    scores_filt = [scores_filt[i] for i in sorted_indices]
    classes_filt = [classes_filt[i] for i in sorted_indices]

    # Apply filter_on_detect_nr if needed
    if params.filter_on_detect_nr > 0:
        boxes_filt = boxes_filt[:params.filter_on_detect_nr]
        scores_filt = scores_filt[:params.filter_on_detect_nr]
        classes_filt = classes_filt[:params.filter_on_detect_nr]

    num_detections_filt = len(boxes_filt)

    return boxes_filt, scores_filt, classes_filt, num_detections_filt


# GET OBJECT CENTER FOR TRACKING
def get_object_center_for_tracking(boxes_filt, params):

    if not boxes_filt:
        object_center = [-1, -1]
        detected = False
    else:
        box = boxes_filt[0]
        startY = int(box[0] * params.h_origin)
        startX = int(box[1] * params.w_origin)
        endY   = int(box[2] * params.h_origin)
        endX   = int(box[3] * params.w_origin)

        Y_center = int((endY-startY)/2) + startY
        X_center = int((endX-startX)/2) + startX

        object_center = [X_center, Y_center]
        detected = True

    return object_center, detected