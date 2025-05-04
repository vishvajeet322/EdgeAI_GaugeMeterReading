import tensorflow as tf
import numpy as np
import cv2
import glob

# Step 1: Load saved model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/nirbhays/sem2_V/_V2/yolov5/yolov5_tf_model")  # or from_keras_model()

# Step 2: Enable full INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Step 3: Define representative dataset
def representative_data_gen():
    image_paths = glob.glob("rep_data/*.jpg")[:100]  # Use ~100 images
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (input_width, input_height))  # Match your model’s input
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

converter.representative_dataset = representative_data_gen

# Step 4: Apply int8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Step 5: Convert
quant_model = converter.convert()

# Step 6: Save
with open("gauge_reader_int8.tflite", "wb") as f:
    f.write(quant_model)

print("✅ Fully quantized model saved as gauge_reader_int8.tflite")
