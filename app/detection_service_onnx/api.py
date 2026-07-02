from fastapi import FastAPI, Body
import detection_service_onnx.detection as dt
from pydantic import BaseModel
import numpy as np

class DetectionParams(BaseModel):
    h_origin: int = 320
    w_origin: int = 320
    h_input_frame: int = 320
    w_input_frame: int = 320
    c_input_frame: int = 3
    class_filter: int = 37
    classif_th: float = 0.2
    filter_on_detect_nr: int = 1

# load CV model
MODEL_PATH = "detection_service_onnx/model_onnx/model.onnx"
model_infer_fcn = dt.ONNXModel(MODEL_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Detection service is running."}

@app.post("/detect")
def detect(
    frame_bytes: bytes = Body(...),
    h_origin: int = 320,
    w_origin: int = 320,
    h_input_frame: int = 320,
    w_input_frame: int = 320,
    c_input_frame: int = 3,
    class_filter: int = 37,
    classif_th: float = 0.2,
    filter_on_detect_nr: int = 1
    ):

    params = DetectionParams(
        h_origin = h_origin,
        w_origin = w_origin,
        h_input_frame = h_input_frame,
        w_input_frame = w_input_frame,
        c_input_frame = c_input_frame,
        class_filter = class_filter,
        classif_th = classif_th,
        filter_on_detect_nr = filter_on_detect_nr
    )

    # reconstruct numpy array from raw bytes
    frame = np.frombuffer(
        frame_bytes,
        dtype=np.uint8
    )

    # restore image shape
    frame = frame.reshape(
        params.h_input_frame,
        params.w_input_frame,
        params.c_input_frame
    )

    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)

    results = dt.object_detection_lite_fcn(model_infer_fcn, frame, params)

    return {
        "detected": results["detected"],
        "class": results["class"],
        "box": results["box"],
        "score": results["score"],
        "object_center": results["object_center"]
    }
