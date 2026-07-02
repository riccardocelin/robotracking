import onnxruntime as ort
import numpy as np
import time

class ONNXModel:

    def __init__(self, model_path):

        self.session = ort.InferenceSession(model_path)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, input_tensor):

        # Ensure correct dtype (critical)
        #input_tensor = input_tensor.astype(np.uint8)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )

        # Return TF-like dict (this is what your code expects)
        return dict(zip(self.output_names, outputs))


def object_detection_lite_fcn(model_infer_fcn, prepro_frame, params):

    # Run inference
    outputs = model_infer_fcn(
        input_tensor=prepro_frame
    )

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

        valid_scores = scores[mask]

        best_score_idx = np.argmax(valid_scores) # best score for selected class

        score = valid_scores[best_score_idx]

        print(f"target_class: {target_class}")
        print(f"best score idx: {best_score_idx}")
        print(f"best score: {score}")
        print(f"params.classif_th: {params.classif_th}")

        if score >= params.classif_th:

            print(f"sono entrato in score >= params.classif_th")

            detected = True
            box = boxes[best_score_idx]

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