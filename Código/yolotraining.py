import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE = r"C:\CODIGOSAR\COCO Car Damage (Projeto SAR)"
YOLO_DIR = r"C:\yolo_car_damage"

def converter_coco_para_yolo(json_path, img_src, img_dst, label_dst):
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)
    with open(json_path) as f:
        data = json.load(f)
    anns_por_imagem = {}
    for ann in data['annotations']:
        anns_por_imagem.setdefault(ann['image_id'], []).append(ann)
    for img in data['images']:
        src = os.path.join(img_src, img['file_name'])
        if not os.path.exists(src):
            continue
        shutil.copy(src, os.path.join(img_dst, img['file_name']))
        W, H = img['width'], img['height']
        label_file = os.path.join(label_dst, img['file_name'].replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            for ann in anns_por_imagem.get(img['id'], []):
                x, y, w, h = ann['bbox']
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}\n")

converter_coco_para_yolo(
    os.path.join(BASE, "train", "COCO_train_annos.json"),
    os.path.join(BASE, "train"),
    os.path.join(YOLO_DIR, "images", "train"),
    os.path.join(YOLO_DIR, "labels", "train")
)
converter_coco_para_yolo(
    os.path.join(BASE, "val", "COCO_val_annos.json"),
    os.path.join(BASE, "val"),
    os.path.join(YOLO_DIR, "images", "val"),
    os.path.join(YOLO_DIR, "labels", "val")
)

# YAML com barras normais
yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
with open(yaml_path, 'w') as f:
    f.write("path: C:/yolo_car_damage\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 1\n")
    f.write("names: ['damage']\n")

print("Dataset convertido! Iniciando treinamento...")

model = YOLO("yolov8n.pt")
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=4,
    name="yolov8_car_damage",
    patience=10
)

print("Treinamento concluído!")