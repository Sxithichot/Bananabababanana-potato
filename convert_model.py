"""Convert best_optimized_banana_model.keras -> models/banana_model.onnx
Usage: python convert_model.py path/to/best_optimized_banana_model.keras
Needs: pip install tensorflow tf2onnx   (only for this one-off step)
"""
import sys, tensorflow as tf, tf2onnx
src = sys.argv[1]
model = tf.keras.models.load_model(src)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="models/banana_model.onnx")
print("saved models/banana_model.onnx")
