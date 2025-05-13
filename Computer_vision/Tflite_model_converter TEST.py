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

tflite_model_name = "ssd_mobilenet_v2_320x320_coco17_tpu-8_TEST"

current_file_path = Path(__file__).resolve()
current_dir = current_file_path.parent

# Set the path to your SavedModel directory
model_folder = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
model_fullfolder = model_folder
saved_model_dir = os.path.join(current_dir, model_fullfolder, "saved_model")

# Converter di TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Imposta l'input come un tensore uint8 con forma fissa
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.uint8]

# Definizione dell'input con forma fissa
converter.experimental_new_converter = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Definizione della forma fissa dell'input
def representative_dataset():
    for _ in range(100):
        yield [tf.random.uniform((1, 320, 320, 3), minval=0, maxval=255, dtype=tf.uint8)]

converter.representative_dataset = representative_dataset

# Converti il modello
tflite_model = converter.convert()

tflite_path = os.path.join(current_dir, model_folder, "TFLite", tflite_model_name + ".tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted and saved to {tflite_path}")
