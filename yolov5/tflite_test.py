import tensorflow as tf
import numpy as np
import cv2

# Path to the TFLite model
tflite_model_path = "gauge_reader_quantized.tflite"

# Path to a sample test image (provide your own image path here!)
test_image_path = "/home/nirbhays/sem2_V/_V2/yolov5/test_dataset_gauge/Dial_Images/Image pointer-20/img_012.jpg"  # <<< UPDATE THIS!!


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model input details: {input_details}")
print(f"✅ Model output details: {output_details}")

# Load and preprocess the input image
input_shape = input_details[0]['shape']
batch_size, channels, img_height, img_width = input_shape

# Load image using OpenCV
img = cv2.imread(test_image_path)
img = cv2.resize(img, (img_width, img_height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = img / 255.0  # Normalize to [0, 1]
img = np.expand_dims(img, axis=0)  # Shape: (1, height, width, channels)

# Transpose from (1, H, W, C) -> (1, C, H, W)
img = np.transpose(img, (0, 3, 1, 2))

# Set the tensor
interpreter.set_tensor(input_details[0]['index'], img)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"✅ Model output shape: {output_data.shape}")
print(f"✅ Sample model output:\n{output_data}")




