import cv2
import numpy as np
import tensorflow as tf
import math
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/nirbhays/sem2_V/_V2/yolov5/gauge_reader_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
is_channel_first = (input_shape[1] == 3)

input_height = input_shape[2] if is_channel_first else input_shape[1]
input_width = input_shape[3] if is_channel_first else input_shape[2]

CLASS_IDS = {
    0: 'center',
    1: 'pointer',
    2: 'max_value_mark',
    3: 'min_value_mark',
    4: 'pointer_tip'
}

def compute_pressure(coords):
    try:
        center = coords[0]
        pointer_tip = coords[4]
        min_mark = coords[3]
        max_mark = coords[2]

        def angle(a, b):
            return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

        theta_pointer = angle(center, pointer_tip)
        theta_min = angle(center, min_mark)
        theta_max = angle(center, max_mark)

        def angle_between(a, b):
            return (a - b + 360) % 360

        adj_angle = angle_between(theta_pointer, theta_min)
        sweep = angle_between(theta_max, theta_min)
        psi = (adj_angle / sweep) * 100
        return round(psi, 2)
    except:
        return None

# def run_inference(frame):
#     resized = cv2.resize(frame, (input_width, input_height))
#     input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)

#     if is_channel_first:
#         input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # NHWC → NCHW

#     interpreter.set_tensor(input_details[0]['index'], input_tensor)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])[0]

#     coords = {}
#     overlay = frame.copy()
#     h_scale = frame.shape[1] / input_width
#     v_scale = frame.shape[0] / input_height

#     for det in output_data:
#         x, y, w, h, conf, *probs = det
#         class_id = int(np.argmax(probs))
#         score = probs[class_id]
#         if score > 0.5:
#             cx = int(x * h_scale)
#             cy = int(y * v_scale)
#             coords[class_id] = (cx, cy)
#             box_w = int(w * h_scale)
#             box_h = int(h * v_scale)
#             top_left = (int(cx - box_w / 2), int(cy - box_h / 2))
#             bottom_right = (int(cx + box_w / 2), int(cy + box_h / 2))

#             cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)
#             cv2.putText(overlay, CLASS_IDS[class_id], (top_left[0], top_left[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#     return overlay


#     # psi = compute_pressure(coords)
#     # if psi is not None:
#     #     cv2.putText(overlay, f"Reading: {psi} psi", (10, 40),
#     #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
#     #     print(f"Reading: {psi} psi")

#     return overlay

def run_inference(frame):
    resized = cv2.resize(frame, (input_width, input_height))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)

    if is_channel_first:
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # NHWC → NCHW

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

    coords = {}
    overlay = frame.copy()
    h_scale = frame.shape[1] / input_width
    v_scale = frame.shape[0] / input_height

    # --- Prepare boxes, scores, class_ids ---
    boxes = []
    scores = []
    class_ids = []
    for det in raw_output:
        x, y, w, h, conf, *probs = det
        class_id = int(np.argmax(probs))
        score = float(probs[class_id])
        if score > 0.5:
            cx = int(x * h_scale)
            cy = int(y * v_scale)
            bw = int(w * h_scale)
            bh = int(h * v_scale)
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            boxes.append([x1, y1, bw, bh])
            scores.append(score)
            class_ids.append(class_id)

    # --- Apply NMS ---
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.45)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        coords[class_id] = (x + w // 2, y + h // 2)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, CLASS_IDS[class_id], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return overlay


# === Video I/O ===
video_input = "/home/nirbhays/sem2_V/_V2/yolov5/gauge_video_resized_640x640.mp4"
video_output = "output/gauge_output.avi"

cap = cv2.VideoCapture(video_input)
if not cap.isOpened():
    raise IOError(f"❌ Failed to open: {video_input}")

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_output, fourcc, 30.0, (640, 640))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # print(frame.shape)
    # frame = cv2.resize(frame, (640, 720))
    output_frame = run_inference(frame)
    out.write(output_frame)

cap.release()
out.release()
print(f"✅ Gauge reading video saved to: {video_output}")
