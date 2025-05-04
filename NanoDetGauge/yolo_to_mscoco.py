import os
import json
import glob
from PIL import Image

# === CONFIG ===
image_dir = '/home/nirbhays/sem2_V/_V2/gauge_yolo_dataset/test/images'
label_dir = '/home/nirbhays/sem2_V/_V2/gauge_yolo_dataset/test/labels'
output_json = '/home/nirbhays/sem2_V/_V2/gauge_yolo_dataset/test/test.json'
class_names = ['center', 'gauge', 'max', 'min', 'tip']  # <-- update if needed

# === COCO Format Containers ===
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories
for i, name in enumerate(class_names):
    coco['categories'].append({
        "id": i,
        "name": name,
        "supercategory": "object"
    })

image_id = 0
ann_id = 0

image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

for idx, image_path in enumerate(image_files):
    file_name = os.path.basename(image_path)
    label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    # Read image size
    with Image.open(image_path) as img:
        width, height = img.size

    # Add image entry
    coco['images'].append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    })

    print(f" [{idx+1}/{len(image_files)}]", end = "\r")

    # Read annotations
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, w, h = map(float, parts)
            box_w = w * width
            box_h = h * height
            x1 = (xc * width) - box_w / 2
            y1 = (yc * height) - box_h / 2

            coco['annotations'].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x1, y1, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            ann_id += 1

    image_id += 1

# Save to JSON
with open(output_json, 'w') as f:
    json.dump(coco, f, indent=4)

print(f"✅ COCO JSON saved to {output_json}")
