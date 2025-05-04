import cv2
import numpy as np
import tensorflow as tf

def integral_decode(dfl, reg_max=16):
    """Converts DFL-regression outputs to (l, t, r, b) format"""
    dfl = dfl.reshape(-1, 4, reg_max + 1)
    softmax = tf.nn.softmax(dfl, axis=-1).numpy()
    proj = np.arange(reg_max + 1, dtype=np.float32)
    dist = np.sum(softmax * proj, axis=-1)
    return dist

def decode_nanodet_output(output, input_size=(320, 320), num_classes=5, reg_max=7, conf_thresh=0.4, nms_thresh=0.5):
    output = output[0]  # (2100, cls + 4*(reg_max+1))
    cls_logits = output[:, :num_classes]
    dfl_preds = output[:, num_classes:]  # (2100, 4*(reg_max+1))

    # Get scores and class indices
    cls_scores = tf.sigmoid(cls_logits).numpy()
    class_ids = np.argmax(cls_scores, axis=1)
    confidences = np.max(cls_scores, axis=1)

    mask = confidences > conf_thresh
    if np.sum(mask) == 0:
        return []

    class_ids = class_ids[mask]
    scores = confidences[mask]
    dfl_preds = dfl_preds[mask]
    decoded_dist = integral_decode(dfl_preds, reg_max)

    # Anchor centers (grid points)
    featmap_strides = [8, 16, 32]
    anchor_points = []
    for stride in featmap_strides:
        h, w = input_size[0] // stride, input_size[1] // stride
        for y in range(h):
            for x in range(w):
                anchor_points.append([x * stride + stride / 2, y * stride + stride / 2])
    anchor_points = np.array(anchor_points, dtype=np.float32)[mask]

    # Decode bbox: center -> [x1, y1, x2, y2]
    cx, cy = anchor_points[:, 0], anchor_points[:, 1]
    l, t, r, b = decoded_dist[:, 0], decoded_dist[:, 1], decoded_dist[:, 2], decoded_dist[:, 3]
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    # Run NMS
    selected = tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=nms_thresh)
    results = []
    for i in selected.numpy():
        results.append([int(class_ids[i]), float(scores[i]), *boxes[i]])

    return results

def draw_highest_score_per_class(detections, image_path, class_names=None, save_path="output.jpg"):
    """
    detections: list of [class_id, score, x1, y1, x2, y2]
    image_path: path to the input image
    class_names: optional list of class names
    save_path: where to save the image with boxes
    """
    img = cv2.imread(TEST_IMAGE_PATH)
    # print(img.shape)
    orig_img = img.copy()
    img = cv2.resize(img, (img_width, img_height))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    # Group by class and keep only highest-score detection
    best_per_class = {}
    for det in detections:
        cls, score, x1, y1, x2, y2 = det
        if cls not in best_per_class or score > best_per_class[cls][0]:
            best_per_class[cls] = (score, (int(x1), int(y1), int(x2), int(y2)))

    # Draw boxes
    for cls, (score, (x1, y1, x2, y2)) in best_per_class.items():
        label = class_names[cls] if class_names else f"Class {cls}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save
    cv2.imwrite(save_path, image)
    print(f"✅ Saved image with detections to: {save_path}")

# Parameters
TFLITE_MODEL_PATH = "gauge_reader_quantized.tflite"
TEST_IMAGE_PATH = "/home/nirbhays/sem2_V/_V2/gauge_yolo_dataset/test/images/0fa016d9ff6a40a5a4e38fd6437c9542_jpeg_jpg.rf.8a8a41ffccad0df36e4d9af412900dd6.jpg"  # <<< Update this
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
# print(img.shape)
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

detections = decode_nanodet_output(output_data, input_size=(img_height, img_width), num_classes=5)

# for det in detections:
#     cls, score, x1, y1, x2, y2 = det
#     print(f"[{cls}] Score: {score:.2f}, Box: {int(x1)},{int(y1)},{int(x2)},{int(y2)}")

class_names = ['center', 'gauge', 'max', 'min', 'tip']
draw_highest_score_per_class(detections, TEST_IMAGE_PATH, class_names, save_path="boxed.jpg")
