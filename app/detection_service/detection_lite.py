import tflite_runtime.interpreter as tflite
from pathlib import Path
import numpy as np
import time

def load_lite_model_signature(model_path_in_prj):
    
    print("-- computer vision model loading")
    current_folder = Path(__file__).resolve().parent
    model_full_path = current_folder.parent / model_path_in_prj

    # Load TFLite model
    interpreter = tflite.Interpreter(
        model_path=str(model_full_path / "model.tflite"),
        num_threads=4
    )

    print("signature list:\n")
    print(interpreter.get_signature_list())
    print("\nget_input_details:\n")
    print(interpreter.get_input_details())
    print("\nget_output_details:\n")
    print(interpreter.get_output_details())

    interpreter.allocate_tensors()

    # Get signature runner
    infer = interpreter.get_signature_runner(
        "serving_default"
    )

    return infer


def object_detection_lite_fcn(model_infer_fcn, prepro_frame, params):

    print(prepro_frame.shape)
    print(prepro_frame.dtype)
    
    s = time.time()
    # Run inference
    outputs = model_infer_fcn(
        input_tensor=prepro_frame
    )
    e = time.time()
    infer = e-s
    print(f"inference api: {infer}")

    # Print outputs
    for name, value in outputs.items():
        print(name, value.shape)

    boxes = outputs["detection_boxes"][0]
    scores = outputs["detection_scores"][0]
    classes = outputs["detection_classes"][0]


    target_class = params.class_filter

    mask = classes == target_class


    box = [-1,-1,-1,-1]
    score = -1
    object_center = [-1,-1]
    detected = False
    if np.any(mask):

        print("sono entrato nel any(mask)")

        valid_scores = scores[mask]

        best_score_idx = np.argmax(valid_scores) # best score for selected class

        score = valid_scores[best_score_idx]

        print(f"best score idx: {best_score_idx}")
        print(f"best score: {score}")
        print(f"params.classif_th: {params.classif_th}")

        if score >= params.classif_th:

            print(f"sono entrato in score >= params.classif_th")

            detected = True
            box = boxes[best_score_idx]

            print(f"box: {box}")

            object_center = get_object_center_for_tracking(box, params)

            print(f"box: {box}")
            print(f"score: {score}")
            print(f"best object_center: {object_center}")

    if not isinstance(box, list):
        box = box.tolist()

    response = {
        "detected": bool(detected),
        "class": target_class,
        "box": box,
        "score": float(score),
        "object_center": object_center
    }

    return response


# GET OBJECT CENTER FOR TRACKING
def get_object_center_for_tracking(box, params):

    if np.all(box != -1):
        startY = int(box[0] * params.h_origin)
        startX = int(box[1] * params.w_origin)
        endY   = int(box[2] * params.h_origin)
        endX   = int(box[3] * params.w_origin)

        Y_center = int((endY-startY)/2) + startY
        X_center = int((endX-startX)/2) + startX

        object_center = [X_center, Y_center]

    else:
        object_center = [-1, -1]

    return object_center