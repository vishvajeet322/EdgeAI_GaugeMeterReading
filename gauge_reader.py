import os
import math
import zipfile
import subprocess
import pandas as pd
from PIL import Image

# ------------------------- CONFIGURATION -------------------------
roboflow_zip = 'gauge-analog.v1i.yolov5pytorch.zip'
dataset_folder = 'gauge_yolo_dataset'
yolo_folder = 'yolov5'
corrected_yaml_path = 'gauge_yolo_dataset/data.yaml'

# ------------------------- HELPER FUNCTIONS -------------------------
def unzip_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Dataset extracted to {extract_to}")

def modify_data_yaml(yaml_path):
    corrected_data_yaml = """
train: gauge_yolo_dataset/train/images
val: gauge_yolo_dataset/valid/images
test: gauge_yolo_dataset/test/images

nc: 5
names: ['center', 'pointer', 'max_value_mark', 'min_value_mark', 'pointer_tip']
"""
    with open(yaml_path, "w") as f:
        f.write(corrected_data_yaml)
    print("✅ data.yaml updated successfully.")

def run_training():
    os.chdir(yolo_folder)
    subprocess.run(["python", "train.py", \
                    "--img", "640", \
                    "--batch", "16", \
                    "--epochs", "50", \
                    "--data", "../gauge_yolo_dataset/data.yaml", \
                    "--weights", "yolov5n.pt", \
                    "--name", "gauge_reader_yolov5n"])
    os.chdir("..")

def run_inference():
    os.chdir(yolo_folder)
    subprocess.run(["python", "detect.py", \
                    "--weights", "runs/train/gauge_reader_yolov5n/weights/best.pt", \
                    "--img", "640", \
                    "--source", "../gauge_yolo_dataset/test/images", \
                    "--conf", "0.25", \
                    "--save-txt"])
    os.chdir("..")

def compute_gauge_values(label_dir, image_dir, output_csv):
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    CLASS_IDS = {
        'center': 0,
        'pointer': 1,
        'max_value_mark': 2,
        'min_value_mark': 3,
        'pointer_tip': 4
    }

    def yolo_to_center(bbox, img_w, img_h):
        x_center = float(bbox[1]) * img_w
        y_center = float(bbox[2]) * img_h
        return x_center, y_center

    def angle_between(a, b):
        angle = a - b
        return (angle + 360) % 360

    results = []

    for label_file in label_files:
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        coords = {}
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in CLASS_IDS.values():
                    coords[class_id] = yolo_to_center(parts, img_w, img_h)

        if not all(k in coords for k in [0, 2, 3, 4]):
            continue

        x1, y1 = coords[CLASS_IDS['center']]
        x2, y2 = coords[CLASS_IDS['pointer_tip']]
        xmin, ymin = coords[CLASS_IDS['min_value_mark']]
        xmax, ymax = coords[CLASS_IDS['max_value_mark']]

        theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
        theta_min = math.degrees(math.atan2(ymin - y1, xmin - x1))
        theta_max = math.degrees(math.atan2(ymax - y1, xmax - x1))

        adjusted_theta = angle_between(theta, theta_min)
        gauge_sweep = angle_between(theta_max, theta_min)

        ratio = adjusted_theta / gauge_sweep
        value = ratio * 100  # 0–100 psi

        results.append({
            "image": img_file,
            "theta": round(theta, 2),
            "adjusted_theta": round(adjusted_theta, 2),
            "sweep": round(gauge_sweep, 2),
            "value_predicted": round(value, 2)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved predicted gauge values to {output_csv}")

# ------------------------- MAIN EXECUTION -------------------------
if __name__ == "__main__":

    # 1. Unzip Roboflow dataset (update path to your zip)
    # unzip_dataset(roboflow_zip, dataset_folder)

    # 2. Modify data.yaml
    # modify_data_yaml(corrected_yaml_path)

    # 3. Train YOLOv5 model
    run_training()

    # 4. Run inference
    run_inference()

    # 5. Compute gauge readings
    compute_gauge_values("yolov5/runs/detect/exp/labels", "yolov5/runs/detect/exp", "gauge_predictions.csv")

    print("✅ Completed full pipeline!")


