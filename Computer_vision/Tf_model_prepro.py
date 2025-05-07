import tensorflow as tf
import os
from pathlib import Path

model_folder = "ssd_mobilenet_v2_320x320_coco17_tpu-8"

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

# Set the path to your SavedModel directory
saved_model_dir = os.path.join(current_dir, model_folder, "saved_model")

saved_model_prepro_dir = os.path.join(current_dir, model_folder, "TFLite", "prepro_model_nodynamicinput", "saved_model")

model = tf.saved_model.load(saved_model_dir)
concrete_func = model.signatures['serving_default']

# Update input signature with fixed shape by defining a new function with fixed input shape
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 320, 320, 3], dtype=tf.uint8)])
def fixed_shape_func(input_tensor):
    return concrete_func(input_tensor)

# Save the updated model with the fixed input shape
tf.saved_model.save(model, saved_model_prepro_dir, signatures={'serving_default': fixed_shape_func})