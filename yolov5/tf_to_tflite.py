import tensorflow as tf

# Path to the SavedModel directory
saved_model_dir = "yolov5_tf_model"

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable Select TensorFlow Ops fallback
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # TensorFlow Lite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS           # Also allow full TensorFlow ops like SplitV
]

# (Optional) Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the converted TFLite model
with open("gauge_reader_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully converted to gauge_reader_quantized.tflite with Select TF Ops fallback!")
