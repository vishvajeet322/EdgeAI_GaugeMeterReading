import tensorflow as tf
import numpy as np
import cv2

# Parameters
TFLITE_MODEL_PATH = "gauge_reader_quantized.tflite"
TEST_IMAGE_PATH = "/home/nirbhays/sem2_V/_V2/yolov5/test_dataset_gauge/Dial_Images/Image pointer-20/img_012.jpg"  # <<< Update this
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Class ID to Name Mapping
CLASS_IDS = {
    0: "center",
    1: "guage",
    2: "max_value_mark",
    3: "min_value_mark",
    4: "pointer_tip"
}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

batch_size, channels, img_height, img_width = input_details[0]['shape']

# Load and preprocess image
img = cv2.imread(TEST_IMAGE_PATH)
orig_img = img.copy()
img = cv2.resize(img, (img_width, img_height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)  # (1, H, W, C)
img = np.transpose(img, (0, 3, 1, 2))  # (1, C, H, W)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Post-processing
predictions = np.squeeze(output_data)

# Decode predictions
boxes = []
scores = []
classes = []

for pred in predictions:
    x, y, w, h = pred[0], pred[1], pred[2], pred[3]
    objectness = pred[4]
    class_probs = pred[5:]
    class_id = np.argmax(class_probs)
    confidence = objectness * class_probs[class_id]

    if confidence > CONF_THRESHOLD:
        # Convert from center x,y,w,h to x_min, y_min, x_max, y_max
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(confidence)
        classes.append(class_id)

boxes = np.array(boxes)
scores = np.array(scores)
classes = np.array(classes)

# Perform NMS
nms_indices = tf.image.non_max_suppression(
    boxes=boxes,
    scores=scores,
    max_output_size=50,
    iou_threshold=IOU_THRESHOLD,
    score_threshold=CONF_THRESHOLD
).numpy()

# Draw detections
for idx in nms_indices:
    x_min, y_min, x_max, y_max = boxes[idx]
    
    # Rescale boxes back to original image size
    orig_h, orig_w = orig_img.shape[:2]
    x_min = int((x_min / img_width) * orig_w)
    x_max = int((x_max / img_width) * orig_w)
    y_min = int((y_min / img_height) * orig_h)
    y_max = int((y_max / img_height) * orig_h)

    class_name = CLASS_IDS.get(classes[idx], f"Unknown {classes[idx]}")
    label = f"{class_name}: {scores[idx]:.2f}"
    
    cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(orig_img, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save or Show the final image
output_path = "output_detection_named.jpg"
cv2.imwrite(output_path, orig_img)
print(f"✅ Detection with class names completed. Output saved at {output_path}!")

# Or show it
cv2.imshow("Detection", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
