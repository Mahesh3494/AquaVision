"""
convert_to_onnx.py

Converts the trained Keras (.h5) shrimp disease model to ONNX format.

Originally run in Google Colab (Untitled0.ipynb) -- recovered and committed here
so the training -> ONNX export pipeline is reproducible from the repo.

Why: app.py runs inference using onnxruntime only (no TensorFlow needed at
deploy time). This script is the one-time bridge that produces the .onnx
file app.py loads.
"""

import tf2onnx
import tensorflow as tf
import onnx
import os

# Load model
model = tf.keras.models.load_model('/content/drive/MyDrive/aquavision/best_model.h5')
print("Model loaded successfully")

# Convert to ONNX
# Fixed batch size of 1 -- matches app.py, which always predicts on a single
# uploaded image at a time (no batch prediction needed).
input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32)]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save
onnx.save(onnx_model, '/content/drive/MyDrive/aquavision/shrimp_model.onnx')

size = os.path.getsize('/content/drive/MyDrive/aquavision/shrimp_model.onnx')
print(f"Done! ONNX model size: {size / 1024 / 1024:.1f} MB")