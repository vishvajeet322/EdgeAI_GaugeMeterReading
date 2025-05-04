import torch
from nanodet.model.arch import build_model
from nanodet.util import load_config, Logger, cfg
from nanodet.util.check_point import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.data.transform.warp import warp_boxes
import cv2
import numpy as np
import os

# === USER CONFIG ===
cfg_path = "config/legacy_v0.x_configs/gauge_config.yml"
model_path = "workspace/gauge_nanodet_m_0.5x/gauge_nanodet.ckpt"
image_path = "/home/nirbhays/sem2_V/_V2/gauge_yolo_dataset/test/images/0fa016d9ff6a40a5a4e38fd6437c9542_jpeg_jpg.rf.8a8a41ffccad0df36e4d9af412900dd6.jpg"  # <<< Update this
class_names = ['center', 'gauge', 'max', 'min', 'tip']

# === Load Config and Model ===
load_config(cfg, cfg_path)
logger = Logger(-1, use_tensorboard=False)
model = build_model(cfg.model)
ckpt = torch.load(model_path, map_location="cpu")
load_model_weight(model, ckpt, logger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# === Prepare Image ===
img = cv2.imread(image_path)
orig_h, orig_w = img.shape[:2]
img_input = img.copy()

# Apply test pipeline (resizing, normalization, etc.)
test_pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
data = {"img": img_input, "img_info": {"height": orig_h, "width": orig_w}}
data = test_pipeline({}, data, (320,320))
tensor_img = torch.from_numpy(data["img"]).unsqueeze(0).to(device)

# === Run Inference ===
with torch.no_grad():
    meta = {"img_info": data["img_info"], "warp_matrix": data["warp_matrix"]}
    preds = model(tensor_img)
    results = model.head.post_process(preds, meta)

# === Parse + Visualize ===
result = results[data["img_info"]["id"]]  # dict of class_id -> [[x1, y1, x2, y2, score], ...]
for class_id, bboxes in result.items():
    for box in bboxes:
        x1, y1, x2, y2, score = map(int, box[:4]) + [box[4]]
        if score < 0.3:
            continue
        label = class_names[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

cv2.imwrite("output.jpg", img)
print("✅ Inference complete. Saved as output.jpg.")
