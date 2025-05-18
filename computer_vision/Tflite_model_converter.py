import tensorflow as tf
import numpy as np
import os
import cv2
from pathlib import Path

# notes:
#
# this script load the prepro_saved_model and convert it into a .tflite format for edge deployment.
# the prepro_saved_model is the same model as the downloaded one despite from the fact that the input
# shape of the model in fixed to the expectd size of [1, 320, 320, 3] to avoid conversion issues.
#
# final model to be deployed saved in: .../TFLite/tflite_model_name.tflite

tflite_model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8_nodynamicinput"

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

# Set the path to your SavedModel directory
model_folder = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
model_fullfolder = model_folder + "/TFLite/prepro_model_nodynamicinput"
saved_model_dir = os.path.join(current_dir, model_fullfolder, "saved_model")

# Create the TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

### check before conversion ###
model = tf.saved_model.load(saved_model_dir)
print(list(model.signatures.keys()))  # Check available signatures
# Check the input details of the serving signature
concrete_func = model.signatures['serving_default']
print(concrete_func.structured_input_signature)

# Explicitly set the input shape to (1, 320, 320, 3) to avoid later on issues
converter._input_shapes = {"serving_default_input_tensor": [1, 320, 320, 3]}

# Step 1: Enable optimization (this activates quantization-aware behavior)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Step 2: Provide a representative dataset (from COCO dataset) for full integer quantization
# Replace with samples from your real dataset if possible
COCO_VAL_IMAGES = os.path.join(current_dir, model_folder, "TFLite/coco_subsampl_val2017")  # Folder of real COCO validation images
image_paths = [os.path.join(COCO_VAL_IMAGES, f) for f in os.listdir(COCO_VAL_IMAGES) if f.endswith('.jpg')]

def representative_dataset():
    for path in image_paths[:200]:
        img = cv2.imread(path)
        img = cv2.resize(img, (320, 320))  # Resize to model input size
        img = np.expand_dims(img, axis=0).astype(np.uint8)  # Add batch dim
        yield [img]
converter.representative_dataset = representative_dataset

# Step 3: Set input/output types for Raspberry Pi compatibility
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Step 4: Convert the model
tflite_model = converter.convert()

# Step 5: Save the TFLite model
tflite_path = os.path.join(current_dir, model_folder, "TFLite", tflite_model_name + ".tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_path}")

