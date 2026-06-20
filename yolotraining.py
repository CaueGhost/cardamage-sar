"""
yolotraining.py — Treinamento e avaliação com YOLOv8
Converte o dataset COCO para formato YOLO, treina o modelo e
avalia TODAS as imagens de validação salvando os resultados.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# ── Caminhos relativos ──────────────────────────────────────────────────────────
DIR      = os.path.dirname(os.path.abspath(__file__))
BASE     = os.path.join(DIR, "..", "COCO Car Damage (Projeto SAR)")
YOLO_DIR = os.path.join(DIR, "..", "yolo_dataset")
OUT      = os.path.join(DIR, "..", "resultados_yolo")
os.makedirs(OUT, exist_ok=True)


def converter_coco_para_yolo(json_path, img_src, img_dst, label_dst):
    """Converte anotações do formato COCO JSON para o formato YOLO (txt)."""
    os.makedirs(img_dst,   exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    anns_por_imagem = {}
    for ann in data["annotations"]:
        anns_por_imagem.setdefault(ann["image_id"], []).append(ann)

    print(f"Convertendo {len(data['images'])} imagens de {os.path.basename(img_src)}...")
    for img in tqdm(data["images"], unit="img"):
        src = os.path.join(img_src, img["file_name"])
        if not os.path.exists(src):
            continue
        shutil.copy(src, os.path.join(img_dst, img["file_name"]))

        W, H        = img["width"], img["height"]
        label_file  = os.path.join(label_dst, img["file_name"].replace(".jpg", ".txt"))

        with open(label_file, "w") as f:
            for ann in anns_por_imagem.get(img["id"], []):
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}\n")


# ── Converte treino e validação ─────────────────────────────────────────────────
converter_coco_para_yolo(
    os.path.join(BASE, "train", "COCO_train_annos.json"),
    os.path.join(BASE, "train"),
    os.path.join(YOLO_DIR, "images", "train"),
    os.path.join(YOLO_DIR, "labels", "train"),
)
converter_coco_para_yolo(
    os.path.join(BASE, "val", "COCO_val_annos.json"),
    os.path.join(BASE, "val"),
    os.path.join(YOLO_DIR, "images", "val"),
    os.path.join(YOLO_DIR, "labels", "val"),
)

# ── Cria o arquivo de configuração YAML ────────────────────────────────────────
yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
yolo_dir_unix = YOLO_DIR.replace("\\", "/")
with open(yaml_path, "w") as f:
    f.write(f"path: {yolo_dir_unix}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 1\n")
    f.write("names: ['damage']\n")

# ── Treino ──────────────────────────────────────────────────────────────────────
print("\nIniciando treinamento YOLOv8...")
model = YOLO("yolov8n.pt")
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=4,
    name="yolov8_car_damage",
    patience=10,
    verbose=True,
)
print("Treinamento concluído!")

# ── Avaliação em TODAS as imagens de validação ──────────────────────────────────
print("\nGerando predições para todas as imagens de validação...")

with open(os.path.join(BASE, "val", "COCO_val_annos.json")) as f:
    val_data = json.load(f)

anns_por_img = {}
for ann in val_data["annotations"]:
    anns_por_img.setdefault(ann["image_id"], []).append(ann)

total_reais = 0
total_detec = 0
acertos     = 0

for img_info in tqdm(val_data["images"], desc="Avaliando", unit="img"):
    img_path = os.path.join(BASE, "val", img_info["file_name"])
    if not os.path.exists(img_path):
        continue

    img_pil  = Image.open(img_path).convert("RGB")
    gt_boxes = [ann["bbox"] for ann in anns_por_img.get(img_info["id"], [])]
    n_reais  = len(gt_boxes)
    total_reais += n_reais

    # Predição YOLOv8
    pred    = model.predict(img_path, conf=0.1, verbose=False)[0]
    boxes   = [b for b in pred.boxes if b.conf[0] > 0.1]
    n_detec = len(boxes)
    total_detec += n_detec
    acertos     += min(n_detec, n_reais)

    # ── Salva imagem com anotações ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(img_info["file_name"], fontsize=12, fontweight="bold")

    # Painel esquerdo — Ground truth
    axes[0].imshow(img_pil)
    axes[0].set_title(f"Ground Truth ({n_reais} danos)", color="green", fontweight="bold")
    axes[0].axis("off")
    for bbox in gt_boxes:
        x, y, w, h = bbox
        axes[0].add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))

    # Painel direito — Predição YOLOv8
    axes[1].imshow(img_pil)
    axes[1].set_title(f"YOLOv8 ({n_detec} detecções)", color="goldenrod", fontweight="bold")
    axes[1].axis("off")
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = b.conf[0].item()
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="gold", facecolor="none"))
        axes[1].text(x1, y1 - 4, f"{conf:.2f}", color="gold", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"pred_{img_info['file_name']}"), dpi=120, bbox_inches="tight")
    plt.close()

# ── Resumo final ────────────────────────────────────────────────────────────────
print("\n" + "=" * 45)
print("     RESULTADO FINAL — YOLOv8")
print("=" * 45)
print(f"  Imagens avaliadas    : {len(val_data['images'])}")
print(f"  Danos reais (total)  : {total_reais}")
print(f"  Detecções (conf>0.1) : {total_detec}")
print(f"  Acertos estimados    : {acertos}/{total_reais}")
taxa = acertos / total_reais * 100 if total_reais > 0 else 0
print(f"  Taxa de acerto       : {taxa:.1f}%")
print(f"  Imagens salvas em    : {OUT}")
print("=" * 45)